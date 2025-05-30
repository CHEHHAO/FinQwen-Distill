import os
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:64'


# 1. 模型与分词器路径
model_name = "/data/mengao/models/Qwen/Qwen2.5-7B-Instruct"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True
)

# 2. 加载模型
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    pad_token_id=tokenizer.pad_token_id,
    trust_remote_code=True
).to(device)  # 将模型移动到 GPU 上

model.eval()  # 设置模型为评估模式


# 3. 定义测试用例 Prompt
prompt = """
<|im_start|>[INST] <<SYS>>
You are a named-entity recognition assistant. Identify all company names in the following title and text, and output an array of strings, each string being one company name.
<</SYS>>
"title": "Quest Partners LLC Decreases Stock Position in Core Laboratories Inc. (NYSE:CLB)",
"text": "Quest Partners LLC cut its holdings in Core Laboratories Inc. ( NYSE:CLB – Free Report ) by 40.4% during the 3rd quarter, according to its most recent 13F filing with the Securities & Exchange Commission. The institutional investor owned 9,653 shares of the oil and gas company’s stock after selling 6,531 shares during the period. Quest Partners LLC’s holdings in Core Laboratories were worth $179,000 as of its most recent SEC filing. Other institutional investors and hedge funds have also made changes to their positions in the company. Headlands Technologies LLC purchased a new stake in shares of Core Laboratories in the 2nd quarter worth about $36,000. nVerses Capital LLC purchased a new stake in shares of Core Laboratories in the 3rd quarter worth about $48,000. GAMMA Investing LLC boosted its position in shares of Core Laboratories by 23.9% in the 2nd quarter. GAMMA Investing LLC now owns 4,425 shares of the oil and gas company’s stock worth $90,000 after buying an additional 855 shares in the last quarter. Northwestern Mutual Wealth Management Co. lifted its holdings in Core Laboratories by 11.9% in the 2nd quarter. Northwestern Mutual Wealth Management Co. now owns 5,636 shares of the oil and gas company’s stock valued at $114,000 after purchasing an additional 600 shares in the last quarter. Finally, Signaturefd LLC lifted its holdings in Core Laboratories by 12.8% in the 2nd quarter. Signaturefd LLC now owns 6,776 shares of the oil and gas company’s stock valued at $137,000 after purchasing an additional 767 shares in the last quarter. 97.81% of the stock is currently owned by hedge funds and other institutional investors. Analysts Set New Price Targets A number of research firms have issued reports on CLB. Citigroup reduced their target price on shares of Core Laboratories from $15.00 to $14.00 and set a “sell” rating on the stock in a report on Thursday, October 31st. StockNews.com upgraded shares of Core Laboratories from a “sell” rating to a “hold” rating in a report on Thursday, November 7th. Two research analysts have rated the stock with a sell rating and three have given a hold rating to the stock. According to MarketBeat.com, the stock has a consensus rating of “Hold” and a consensus price target of $17.00. Core Laboratories Price Performance NYSE:CLB opened at $20.43 on Thursday. The firm has a market cap of $959.19 million, a price-to-earnings ratio of 31.52, a price-to-earnings-growth ratio of 1.40 and a beta of 2.35. Core Laboratories Inc. has a 12-month low of $13.82 and a 12-month high of $25.13. The company has a current ratio of 2.48, a quick ratio of 1.79 and a debt-to-equity ratio of 0.55. The company’s 50-day moving average is $19.38 and its two-hundred day moving average is $19.52. Core Laboratories ( NYSE:CLB – Get Free Report ) last announced its quarterly earnings data on Wednesday, October 23rd. The oil and gas company reported $0.25 earnings per share for the quarter, topping the consensus estimate of $0.21 by $0.04. The business had revenue of $134.40 million for the quarter, compared to analyst estimates of $134.16 million. Core Laboratories had a net margin of 5.83% and a return on equity of 15.84%. The company’s revenue was up 7.2% on a year-over-year basis. During the same period in the prior year, the firm posted $0.22 EPS. Research analysts expect that Core Laboratories Inc. will post 0.8 EPS for the current year. Core Laboratories Announces Dividend The business also recently disclosed a quarterly dividend, which was paid on Monday, November 25th. Stockholders of record on Monday, November 4th were issued a dividend of $0.01 per share. The ex-dividend date of this dividend was Monday, November 4th. This represents a $0.04 annualized dividend and a dividend yield of 0.20%. Core Laboratories’s payout ratio is currently 6.15%. Core Laboratories Profile ( Free Report ) Core Laboratories Inc provides reservoir description and production enhancement services and products to the oil and gas industry in the United States, and internationally. It operates through Reservoir Description and Production Enhancement segments. The Reservoir Description segment includes the characterization of petroleum reservoir rock and reservoir fluid samples to enhance production and improve recovery of crude oil and gas from its clients' reservoirs. See Also Receive News & Ratings for Core Laboratories Daily - Enter your email address below to receive a concise daily summary of the latest news and analysts' ratings for Core Laboratories and related companies with MarketBeat.com's FREE daily email newsletter .",
[/INST]
<|im_end|>
"""

with torch.no_grad():  # 禁用梯度计算以节省内存
    # 4. 编码输入
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048,  # 根据模型的最大输入长度调整
    ).to(device)  # 将输入移动到 GPU 上

    # 5. 生成输出
    outputs = model.generate(
        **inputs.to(device),
        max_new_tokens=256,  # 设置生成的最大新令牌数
        do_sample=True,    # 为了测试一致性，使用贪心模式
        eos_token_id=tokenizer.eos_token_id,
    )

# 6. 解码输出
print("Output text:", tokenizer.decode(outputs[0], skip_special_tokens=True))

torch.cuda.empty_cache()  # 清理 GPU 内存

