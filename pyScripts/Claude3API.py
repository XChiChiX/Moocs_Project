import anthropic

api_key = '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'

client = anthropic.Anthropic( api_key = api_key)

def get_data():
    with open('./data/clips/textAndTime_1.txt', 'r', encoding="UTF-8") as file:
        return file.read()

def get_res(client, prompt):

    return client.messages.create(
        model="claude-3-opus-20240229",
        # model="claude-3-sonnet-20240229", # 模型型號
        max_tokens=4096, # 選用，回傳token的最大長度，避免爆預算
        messages=[
            {"role": "user", "content": prompt}
        ]
    ).content[0].text

def claude3api():
    claudeRes = get_res(client,
        f"""{get_data()}
        檔案內有每一句話的開始時間與結束時間，幫我根據內容分成三段，整理出每段摘要(300字)並根據每段內容提出三個問題，以及每段開始時間，請依照以下格式輸出(第一行 每段開始時間 用空白隔開，第二行 第一段摘要，第三~五行 第一段落的問題，第六行 第二段摘要 以此類推)
        """
    )
    with open("./data/clips/ClaudeRes.txt", "w", encoding="UTF-8") as output_file:
        output_file.write(claudeRes)
    print(claudeRes)
        
    print("Claude3API completed...")

# claude3api()