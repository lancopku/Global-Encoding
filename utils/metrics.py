'''
 @Date  : 2017/12/18
 @Author: Shuming Ma
 @mail  : shumingma@pku.edu.cn 
 @homepage: shumingma.com
'''
import pyrouge
import codecs
import os
import logging

def bleu(reference, candidate, log_path, print_log, config):
    ref_file = log_path+'reference.txt'
    cand_file = log_path+'candidate.txt'
    with codecs.open(ref_file, 'w', 'utf-8') as f:
        for s in reference:
            if not config.char:
                f.write(" ".join(s)+'\n')
            else:
                f.write("".join(s) + '\n')
    with codecs.open(cand_file, 'w', 'utf-8') as f:
        for s in candidate:
            if not config.char:
                f.write(" ".join(s).strip()+'\n')
            else:
                f.write("".join(s).strip() + '\n')

    if config.refF != '':
        ref_file = config.refF

    temp = log_path + "result.txt"
    command = "perl script/multi-bleu.perl " + ref_file + "<" + cand_file + "> " + temp
    os.system(command)
    with open(temp) as ft:
        result = ft.read()
    os.remove(temp)
    print_log(result)

    return float(result.split()[2][:-1])


def rouge(reference, candidate, log_path, print_log, config):
    '''print(len(reference), len(candidate), candidate[:5])
    len_sum = 0
    for i in range(len(reference)):
        len_sum += len(reference[i].split())
    print(len_sum/len(reference))'''
    assert len(reference) == len(candidate)

    ref_dir = log_path + 'reference/'
    cand_dir = log_path + 'candidate/'
    if not os.path.exists(ref_dir):
        os.mkdir(ref_dir)
    if not os.path.exists(cand_dir):
        os.mkdir(cand_dir)

    for i in range(len(reference)):
        with codecs.open(ref_dir+"%06d_reference.txt" % i, 'w', 'utf-8') as f:
            f.write(" ".join(reference[i]).replace(' <\s> ', '\n') + '\n')
        with codecs.open(cand_dir+"%06d_candidate.txt" % i, 'w', 'utf-8') as f:
            f.write(" ".join(candidate[i]).replace(' <\s> ', '\n').replace('<unk>', 'UNK') + '\n')

    r = pyrouge.Rouge155()
    r.model_filename_pattern = '#ID#_reference.txt'
    r.system_filename_pattern = '(\d+)_candidate.txt'
    r.model_dir = ref_dir
    r.system_dir = cand_dir
    logging.getLogger('global').setLevel(logging.WARNING)
    rouge_results = r.convert_and_evaluate()
    scores = r.output_to_dict(rouge_results)
    recall = [round(scores["rouge_1_recall"] * 100, 2),
              round(scores["rouge_2_recall"] * 100, 2),
              round(scores["rouge_l_recall"] * 100, 2)]
    precision = [round(scores["rouge_1_precision"] * 100, 2),
                 round(scores["rouge_2_precision"] * 100, 2),
                 round(scores["rouge_l_precision"] * 100, 2)]
    f_score = [round(scores["rouge_1_f_score"] * 100, 2),
               round(scores["rouge_2_f_score"] * 100, 2),
               round(scores["rouge_l_f_score"] * 100, 2)]
    print_log("F_measure: %s Recall: %s Precision: %s\n"
              % (str(f_score), str(recall), str(precision)))

    return f_score[:], recall[:], precision[:]

if __name__ == "__main__":
    with codecs.open('/home/linjunyang/giga/valid.src', 'r', 'utf-8') as fc, codecs.open('/home/linjunyang/giga/valid.tgt', 'r', 'utf-8') as fr:
        cand, ref = fc.readlines(), fr.readlines()
        ref_ = []
        for r in ref:
            ref_.append(r.split())
        can = []
        for c in cand:
            can.append(c.split()[:9])
    print(rouge(ref_, can, log_path='/home/linjunyang/'))
    '''with codecs.open('/home/linjunyang/giga/train.src', 'r', 'utf-8') as fc, codecs.open('/home/linjunyang/lcsts/train.tgt', 'r', 'utf-8') as fr, codecs.open('/home/linjunyang/giga/train.src', 'r', 'utf-8') as gc, codecs.open('/home/linjunyang/giga/train.tgt', 'r', 'utf-8') as gr:
        cand, ref, gcand, gref = fc.readlines(), fr.readlines(), gc.readlines(), gr.readlines()
    lsrc, ltgt, giga_src, giga_tgt = [], [], [], []
    for i in range(len(ref)):
        lsrc.append(len(cand[i].split()))
        ltgt.append(len(ref[i].split()))
        giga_src.append(len(gcand[i].split()))
        giga_tgt.append(len(gref[i].split()))
    print(min(lsrc), max(lsrc), sum(lsrc)/len(lsrc), min(ltgt), max(ltgt), sum(ltgt)/len(ltgt))
    print(min(giga_src), max(giga_src), sum(giga_src)/len(giga_src), min(giga_tgt), max(giga_tgt), sum(giga_tgt)/len(giga_tgt))'''
