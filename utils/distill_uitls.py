import torch
import torch.nn as nn
import torch.nn.functional as F
import deepspeed
from dataclasses import dataclass
from trl import SFTTrainer, SFTConfig
from trl.trainer.utils import DataCollatorForChatLM, empty_cache
from typing import Optional, Union
from transformers import GenerationConfig, PreTrainedModel


@dataclass
class DistillConfig(SFTConfig):
    """
    Configuration class for distillation.
    Args:
        temperature (`float`, *optional*, defaults to `0.9`):
            Temperature for sampling. The higher the temperature, the more random the completions.
        alpha (`float`, *optional*, defaults to `0.5`):
            Alpha parameter that controls the importance of the KL divergence term in the loss.
        max_new_tokens (`int`, *optional*, defaults to `1024`):
            Maximum number of tokens to generate per completion.
    """

    temperature: float = 0.9
    alpha: float = 1
    max_new_tokens: int = 1024

    def __post_init__(self):
        super().__post_init__()
        # check alpha is in the range [0, 1]
        if self.alpha < 0 or self.alpha > 1:
            raise ValueError(f"Alpha must be in the range [0, 1], but got {self.alpha}.")


class DistillTrainer(SFTTrainer):
    def __init__(
        self, 
        teacher_model: Union[PreTrainedModel, nn.Module, str],
        args: Optional[DistillConfig] = None,
        *sft_args, 
        **kwargs):

        args.remove_unused_columns = False
        kwargs["data_collator"] = DataCollatorForChatLM(tokenizer=kwargs["tokenizer"], max_length=args.max_seq_length)
        super().__init__(*sft_args, args=args, **kwargs)
        self.teacher_model = self.accelerator.prepare_model(teacher_model, evaluation_mode=True)
        self.alpha = args.alpha
        self.temperature = args.temperature

        if self.is_deepspeed_enabled:
            self.teacher_model = self._prepare_deepspeed(teacher_model)
        else:
            self.teacher_model = self.accelerator.prepare_model(teacher_model, evaluation_mode=True)

        self.generation_config = GenerationConfig(
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            do_sample=True,
            top_k=0,
            use_cache=False if args.gradient_checkpointing else True,
        )


    def compute_loss(self, model, inputs, return_outputs=False):
        # compute the loss for the student model
        outputs_student = model(
            input_ids = inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs["labels"],
        )
        student_loss = outputs_student.loss

        # compute the loss for the teacher model
        self.teacher_model.eval()
        with torch.no_grad():
            outputs_teacher = self.teacher_model(
                input_ids = inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )

        # slice the logits for the generated tokens using the inputs[prompts] lengths
        prompt_lengths = inputs["prompts"].shape[1]
        shifted_student_logits = outputs_student.logits[:, prompt_lengths - 1:-1, :]
        shifted_teacher_logits = outputs_teacher.logits[:, prompt_lengths - 1:-1, :]
        # compute the KL divergence loss
        loss_function = nn.KLDivLoss(reduction="batchmean")
        loss_logits = (
            loss_function(
            F.log_softmax(shifted_student_logits / self.temperature, dim=-1),
            F.softmax(shifted_teacher_logits / self.temperature, dim=-1)
            )
            * (self.temperature**2)
        )
        # Return weights for the two losses
        loss = self.alpha * student_loss + (1 - self.alpha) * loss_logits
        empty_cache()
        
        return (loss, outputs_student) if return_outputs else loss
    

    def _prepare_deepspeed(self, model):
        