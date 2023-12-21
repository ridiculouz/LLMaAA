from collections import Counter

def get_entities(tag_seq: list, id2tag: dict=None):
    """
    id2tag is None means that tag_seq is in 'BMES-{type}' format.
    """
    if id2tag is not None:
        tag_seq = [id2tag[id] for id in tag_seq]
    tag_seq.append('O')

    chunks = []
    chunk = ['', -1, -1]
    for index, tag in enumerate(tag_seq):
        # check for valid entity before each visit
        if chunk[2] != -1:
            chunks.append(tuple(chunk))
            chunk = ['', -1, -1]
        if tag.startswith('S'):
            chunk[1] = index
            chunk[2] = index
            chunk[0] = tag.split('-')[1]
        elif tag.startswith('B'):
            chunk[1] = index
            chunk[0] = tag.split('-')[1]
        elif tag.startswith('M') and chunk[1] != -1:
            _type = tag.split('-')[1]
            if _type != chunk[0]:
                chunk[1] = -1
        elif tag.startswith('E') and chunk[1] != -1:
            _type = tag.split('-')[1]
            if _type != chunk[0]:
                chunk[1] = -1
            else:
                chunk[2] = index
        else:
            # cases: 
            #   tag == 'O'
            #   tag == 'M/E-{}' and chunk[1] == -1
            # tolerate for missing [E]
            if chunk[1] != -1:
                chunk[2] = index - 1
                if chunk[2] >= chunk[1]:
                    chunks.append(tuple(chunk))
                chunk = ['', -1, -1]      
    return chunks


class MetricForNer(object):
    def __init__(self, id2tag):
        self.id2tag = id2tag
        self.reset()

    def reset(self):
        self.actual = []
        self.founds = []
        self.rights = []
    
    def __compute_f1(self, total, found, right):
        """
        Compute micro f1.
        """
        recall = .0 if total == 0 else (right / total)
        precision = .0 if found == 0 else (right / found)
        f1 = .0 if recall + precision == 0 else (2 * precision * recall) / (precision + recall)
        return f1, recall, precision

    def stats(self):
        class_info = {}
        actual_counter = Counter([e[0] for e in self.actual])
        founds_counter = Counter([e[0] for e in self.founds])
        rights_counter = Counter([e[0] for e in self.rights])
        for type_, count in actual_counter.items():
            total = count
            found = founds_counter.get(type_, 0)
            right = rights_counter.get(type_, 0)
            f1, recall, precision = self.__compute_f1(total, found, right)
            class_info[type_] = {'precision': precision, 'recall': recall, 'f1': f1}
        total = len(self.actual)
        found = len(self.founds)
        right = len(self.rights)
        f1, recall, precision = self.__compute_f1(total, found, right)
        overall = {'precision': precision, 'recall': recall, 'f1': f1}
        return f1, overall, class_info

    def update(self, gold_seqs, pred_seqs):
        """
        Accept batch of inputs.
        Example:
            >>> gold_seqs = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
            >>> pred_seqs = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        """
        for gold, pred in zip(gold_seqs, pred_seqs):
            gold_entities = get_entities(gold, self.id2tag)
            pred_entities = get_entities(pred, self.id2tag)
            self.actual.extend(gold_entities)
            self.founds.extend(pred_entities)
            self.rights.extend([e for e in pred_entities if e in gold_entities])

class MetricForRe(object):
    def __init__(self):
        """
        Always assume that id == 0 stands for NA/Other/etc. relations.
        """
        self.reset()
    
    def reset(self):
        self.true_positve = 0
        self.true_negative = 0
        self.false_positve = 0
        self.false_negative = 0
    
    def __compute_f1(self):
        """
        Compute micro f1.
        """
        tp, tn, fp, fn = self.true_positve, self.true_negative, self.false_positve, self.false_negative
        recall = .0 if (tp + fn) == 0 else (tp / (tp + fn))
        precision = .0 if (tp + fp) == 0 else (tp / (tp + fp))
        f1 = .0 if recall + precision == 0 else (2 * precision * recall) / (precision + recall)
        return f1, recall, precision
    
    def __compute_acc(self):
        tp, tn, fp, fn = self.true_positve, self.true_negative, self.false_positve, self.false_negative
        return ((tp + tn) / (tp + tn + fp + fn))

    def stats(self):
        f1, recall, precision = self.__compute_f1()
        acc = self.__compute_acc()
        overall = {'precision': precision, 'recall': recall, 'f1': f1, 'acc': acc}
        return f1, overall

    def update(self, gold_labels, pred_labels):
        """
        Accept batch of inputs.
        Example:
            >>> gold_labels = [0, 0, 1, 2, 3]
            >>> pred_labels = [0, 2, 1, 2, 1]
        """
        for gold, pred in zip(gold_labels, pred_labels):
            if pred == 0:
                if gold == pred:
                    self.true_negative += 1
                else:
                    self.false_negative += 1
            else:
                if gold == pred:
                    self.true_positve += 1
                else:
                    self.false_positve += 1