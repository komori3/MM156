import os
import yaml
import math
from collections import defaultdict


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUBMISSIONS_DIR = os.path.join(ROOT_DIR, 'submissions')


def show_standings(tag_to_total_score, tag_to_bests, tag_to_uniques):

    max_tag_length = len('submission')
    max_score_length = len('score')
    max_bests_length = len('bests')

    list_total_score_to_tag = []
    for tag, total_score in tag_to_total_score.items():
        list_total_score_to_tag.append((total_score, tag))
        max_tag_length = max(max_tag_length, len(tag))
        max_score_length = max(max_score_length, len(str(total_score)))
        max_bests_length = max(max_bests_length, len(str(tag_to_bests[tag])))

    list_total_score_to_tag.sort()

    space_tag = max_tag_length - len('submission') + 4
    space_score = max_score_length - len('score') + 4
    space_bests = max_bests_length - len('bests') + 4

    print('submission' + (' ' * space_tag) + 'score' + (' ' * space_score) + 'bests' + (' ' * space_bests) + 'uniques')
    print('-' * 100)
    for total_score, tag in list_total_score_to_tag:
        bests = tag_to_bests[tag]
        uniques = tag_to_uniques[tag]
        space_tag = max_tag_length - len(tag) + 4
        space_score = max_score_length - len(str(total_score)) + 4
        space_bests = max_bests_length - len(str(bests)) + 4
        print(tag + (' ' * space_tag) + str(total_score) + (' ' * space_score) + str(bests) + (' ' * space_bests) + str(uniques))


def load_scores(f):
    return list(map(lambda x : {'Seed': int(x[0]), 'Score': float(x[1])}, [line.split('=') for line in str(f.read()).split('\n') if not line == '']))


if __name__ == "__main__":

    tag_to_results = {}
    for tag in os.listdir(SUBMISSIONS_DIR):
        if not os.path.isdir(os.path.join(SUBMISSIONS_DIR, tag)):
            continue
        results_file = os.path.join(SUBMISSIONS_DIR, tag, 'scores.txt')
        with open(results_file) as f:
            tag_to_results[tag] = load_scores(f)

    seed_to_best_score = defaultdict(lambda: -1.0)
    seed_to_unique_tag = defaultdict(lambda: '')
    for tag, results in tag_to_results.items():
        for result in results:
            seed, score = result['Seed'], result['Score']
            if score <= -0.5: continue
            if score == seed_to_best_score[seed]:
                seed_to_unique_tag[seed] = ''
            if seed_to_best_score[seed] < score:
                seed_to_best_score[seed] = score
                seed_to_unique_tag[seed] = tag

    tag_to_total_score = defaultdict(lambda: 0.0)
    tag_to_bests = defaultdict(lambda: 0)
    tag_to_uniques = defaultdict(lambda: 0)

    for tag, results in tag_to_results.items():
        for result in results:
            seed, score = result['Seed'], result['Score']
            if score <= -0.5: continue
            if score == seed_to_best_score[seed]:
                tag_to_bests[tag] += 1
            if score <= 0: continue
            tag_to_total_score[tag] += score / seed_to_best_score[seed]
            # tag_to_total_score[tag] += math.log(result['Score'])
            # tag_to_total_score[tag] += score
        # tag_to_total_score[tag] /= ctr
    
    for seed, tag in seed_to_unique_tag.items():
        if tag == '': continue
        tag_to_uniques[tag] += 1

    show_standings(tag_to_total_score, tag_to_bests, tag_to_uniques)