from typing import List, Iterator, Optional

import torch
from torch.utils.data import Sampler

class BucketBatchSampler(Sampler[List[int]]):
    """
    Agrupa ejemplos de longitudes similares para minimizar el padding en cada batch.
    
    A diferencia de un batch sampler normal, este agrupa muestras que tienen
    una longitud de secuencia similar antes de enviarlas al DataLoader.
    """
    def __init__(
        self, 
        lengths: List[int], 
        batch_size: int, 
        shuffle: bool = True,
        drop_last: bool = False,
        generator: Optional[torch.Generator] = None
    ):
        self.lengths = lengths
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.generator = generator
        self.indices = list(range(len(lengths)))
        
    def __iter__(self) -> Iterator[List[int]]:
        if self.shuffle:
            # Creamos ruido para evitar batches idénticos cada epoch
            if self.generator is not None:
                noise = torch.rand(len(self.lengths), generator=self.generator).tolist()
            else:
                noise = torch.rand(len(self.lengths)).tolist()
            
            # Ruido proporcional (~10%) a la longitud propia para permitir mezcla fina
            noisy_lengths = [l + (n - 0.5) * 0.1 * max(1, l) for l, n in zip(self.lengths, noise)]
            
            # Ordenamos por la longitud ruidosa
            sorted_indices = sorted(self.indices, key=lambda i: noisy_lengths[i])
            
            # Agrupar en batches
            batches = []
            for i in range(0, len(sorted_indices), self.batch_size):
                batch = sorted_indices[i : i + self.batch_size]
                if self.drop_last and len(batch) < self.batch_size:
                    continue
                batches.append(batch)
            
            # Barajar orden de batches (el orden de procesamiento de batches es aleatorio)
            if self.generator is not None:
                batch_order = torch.randperm(len(batches), generator=self.generator).tolist()
            else:
                batch_order = torch.randperm(len(batches)).tolist()
                
            for i in batch_order:
                yield batches[i]
        else:
            # Sin shuffle: ideal para validación si se quiere minimizar padding
            sorted_indices = sorted(self.indices, key=lambda i: self.lengths[i])
            for i in range(0, len(sorted_indices), self.batch_size):
                batch = sorted_indices[i : i + self.batch_size]
                if self.drop_last and len(batch) < self.batch_size:
                    continue
                yield batch
                
    def __len__(self) -> int:
        if self.drop_last:
            return len(self.lengths) // self.batch_size
        else:
            return (len(self.lengths) + self.batch_size - 1) // self.batch_size
