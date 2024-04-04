# def split_file(input_file, lines_per_file):
#     with open(input_file, 'r', encoding="UTF-8") as f:
#         lines = f.readlines()

#     total_lines = len(lines)
#     num_files = total_lines // lines_per_file
#     remainder = total_lines % lines_per_file

#     for i in range(num_files):
#         output_file = f"{input_file}_part_{i+1}.txt"
#         with open(output_file, 'w', encoding="UTF-8") as f:
#             start_index = i * lines_per_file
#             end_index = start_index + lines_per_file
#             f.writelines(lines[start_index:end_index])

#     if remainder > 0:
#         output_file = f"{input_file}_part_{num_files+1}.txt"
#         with open(output_file, 'w', encoding="UTF-8") as f:
#             start_index = num_files * lines_per_file
#             f.writelines(lines[start_index:])

# # 使用示例
# input_file = 'textAndTime.txt'  # 输入文件名
# lines_per_file = 500  # 每个文件的行数
# split_file(input_file, lines_per_file)
# 初始化段落内容

def split_file(input_file):
    with open(input_file, 'r', encoding="UTF-8") as f:
        lines = f.readlines()
    with open("./data/clips/timestamp.txt", 'w', encoding="UTF-8") as f:
        f.writelines(lines[0])
    flag = True
    summary = True
    file_index = 1
    paragraph = []
    for line in lines:
        if line.strip() == "":
            if flag:
                flag = False
            elif summary:
                summary_file = f"./data/clips/summary_part_{file_index}.txt"
                with open(summary_file, 'w', encoding="UTF-8") as f:
                    f.writelines(paragraph)
                summary = False
            else:
                question_file = f"./data/clips/question_part_{file_index}.txt"
                with open(question_file, 'w', encoding="UTF-8") as f:
                    f.writelines(paragraph)
                summary = True
                
                file_index += 1
            
            paragraph = []
        else:
            # 将当前行添加到段落内容中
            paragraph.append(line)
    if paragraph:
        question_file = f"./data/clips/question_part_{file_index}.txt"
        with open(question_file, 'w', encoding="UTF-8") as f:
            f.writelines(paragraph)


    # summary_index = 2
    # for i in range(3):
    #     summary_file = f"summary_part_{i+1}.txt"
    #     question_file = f"question_part_{i+1}.txt"
    #     with open(summary_file, 'w', encoding="UTF-8") as f:
    #         f.writelines(lines[summary_index])
    #     with open(question_file, 'w', encoding="UTF-8") as f:
    #         f.writelines(lines[summary_index + 2:summary_index + 5])

    #     summary_index += 6

def split_txt():
    input_file = './data/clips/ClaudeRes.txt'  # 输入文件名
    split_file(input_file)
    
    print("split_txt completed...")

# split_txt()