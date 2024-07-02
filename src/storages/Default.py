from storages import storage
import click, torch


class Default:
    def __init__(self, dictionary=dict()):
        self.dictionary = dictionary

    def clear(self):
        self.dictionary.clear()

    def append(self,dictionary):
        for k,v in dictionary.items():
            if k in self.dictionary: self.dictionary[k].append(v)
            else: self.dictionary[k] = [v]

    def __len__(self):
        return list(self.dictionary.items()).pop()[1].size(0)

    def stack(self):
        for k,v in self.dictionary.items(): self.dictionary[k] = torch.stack(v)
        return self

    def cat(self):
        for k,v in self.dictionary.items(): self.dictionary[k] = torch.cat(v, dim=0)
        return self

    def flatten(self,*args):
        for k,v in self.dictionary.items(): self.dictionary[k] = v.flatten(*args)
        return self

    def detach(self):
        for k,v in self.dictionary.items(): self.dictionary[k] = v.detach()
        return self

    def __getitem__(self, idx):
        return {k:v[idx] for k,v in self.dictionary.items()}

    def collate_fn(self, data):
        return { k:torch.stack([d[k] for d in data]) for k in data[0].keys() }

    def __str__(self):
        return ", ".join([f"{k}:{v.shape}"for k,v in self.dictionary.items()])

    

@storage.group(invoke_without_command=True)
@click.pass_obj
def default(trainer):
    trainer.set_storage(Default())
