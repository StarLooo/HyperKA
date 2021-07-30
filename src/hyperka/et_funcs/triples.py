# lxy: I think triples.py needn't to be modified
# and I add some comments for readability

# define class Triples to indicate the (h, r, t)-triples
class Triples:
    def __init__(self, triples, ori_triples=None):
        # the set of (h, r, t)-triples
        self.triples = set(triples)
        # the list of (h, r, t)-triples
        self.triple_list = list(self.triples)
        # the num of triple
        self.triples_num = len(triples)

        # the set of heads
        self.heads = set([triple[0] for triple in self.triple_list])
        # the set properties(relations)
        self.props = set([triple[1] for triple in self.triple_list])
        # the set of tails
        self.tails = set([triple[2] for triple in self.triple_list])
        # the set of entities(= heads U tails)
        self.ents = self.heads | self.tails

        # the sorted list of properties
        self.prop_list = list(self.props)
        self.prop_list.sort()

        # the sorted list of entities
        self.ent_list = list(self.ents)
        self.ent_list.sort()

        # I guess ori_triples may be the original triples of this triples
        if ori_triples is None:
            self.ori_triples = None
        else:
            self.ori_triples = set(ori_triples)

        self._generate_related_ents()
        self._generate_triple_dict()
        self._generate_ht()

    def _generate_related_ents(self):
        # out_related_ents_dict is the dict of (key, set(val)) where (key, relation, val) is a triple in this triples
        self.out_related_ents_dict = dict()
        # in_related_ents_dict is the dict of (key, set(val)) where (val, relation, key) is a triple in this triples
        self.in_related_ents_dict = dict()

        for h, r, t in self.triple_list:
            # update out_related_ents_dict
            out_related_ents = self.out_related_ents_dict.get(h, set())
            out_related_ents.add(t)
            self.out_related_ents_dict[h] = out_related_ents
            # update in_related_ents_dict
            in_related_ents = self.in_related_ents_dict.get(t, set())
            in_related_ents.add(h)
            self.in_related_ents_dict[t] = in_related_ents

    def _generate_triple_dict(self):
        # rt_dict is the dict of (key, set((val1, val2))) where (key, val1, val2) is a triple in this triples
        self.rt_dict = dict()
        # hr_dict is the dict of (key, set((val1, val2))) where (val1, val2, key) is a triple in this triples
        self.hr_dict = dict()
        for h, r, t in self.triple_list:
            # update rt_dict
            rt_set = self.rt_dict.get(h, set())
            rt_set.add((r, t))
            self.rt_dict[h] = rt_set
            # update hr_dict
            hr_set = self.hr_dict.get(t, set())
            hr_set.add((h, r))
            self.hr_dict[t] = hr_set

    def _generate_ht(self):
        # ht is the set of (val1, val2) where there exist relation that
        # make triple (val1, relation, val2) is in this triples
        self.ht = set()
        for h, r, t in self.triples:
            self.ht.add((h, t))
