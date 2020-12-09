import os


def rst_analysis(F1, prediction, target, raw_text1, raw_text2, offset1, offset2, model_name, metric_info):
    prediction = prediction[1].gt(0.5).long()
    TP, TN, FP, FN, blk_num = [], [], [], [], len(raw_text1)
    # print(blk_num)
    for idx in range(blk_num):
        # -- insert [BLK] token for raw texts
        tmp_text1 = raw_text1[idx]
        tmp_text1.insert(int(offset1[idx]), '[BLK]')
        tmp_text2 = raw_text2[idx]
        tmp_text2.insert(int(offset2[idx]), '[BLK]')
        # print(prediction[idx], target[idx], prediction[idx] == target[idx], prediction[idx] == 1)
        if prediction[idx] == target[idx] == 1:
            TP.append((' '.join(tmp_text1), ' '.join(tmp_text2)))
        elif prediction[idx] == target[idx] == 0:
            TN.append((' '.join(tmp_text1), ' '.join(tmp_text2)))
        elif prediction[idx] == 1:
            FP.append((' '.join(tmp_text1), ' '.join(tmp_text2)))
        else:
            FN.append((' '.join(tmp_text1), ' '.join(tmp_text2)))
    # -- write files
    para_path = 'results/{}/'.format(model_name)
    if not os.path.exists(para_path):
        os.makedirs(para_path)
    file_path = os.path.join(para_path, '{:.4f}.analyse'.format(F1))
    with open(file_path, 'w') as f:
        f.write('Metrics -- {} \n'.format(metric_info))
        name_lst = ['TP', 'FP', 'FN', 'TN']
        for j, lst in enumerate([TP, FP, FN, TN]):
            name = name_lst[j]
            f.write('-' * 50 + '\n')
            for i, text_tuple in enumerate(lst):
                f.write(name + ' {} of {} \n'.format(i + 1, len(lst)))
                f.write(text_tuple[0] + '\n')
                f.write(text_tuple[1] + '\n')





