import torch
import torch.nn as nn

class Attention(nn.Module):
	"""
	Input:
		- queries Q: ``(batch_size, num_rows_q, embeddin_dim_q)``
		- keys K: 	 ``(batch_size, num_rows_k, embeddin_dim_k)``
 		- values V:  ``(batch_size, num_rows_v, embeddin_dim_v)``
 		- mask:      ``(batch_size, num_rows_v)``

 	Output:
 		- attention: ``(batch_size, num_rows_q, embeddin_dim_q)``

 	:param normalize: ``bool``, optional (default: ``True``)
 		If true, masked softmax is applied. Otherwise, unnormalized 
 		attention weights are returned.
	"""
	def __init__(self, 
				 normalize: bool = True) -> None:
		super().__init__()
		self.normalize = normalize

	@overrides
	def forward(self, 
				queries: torch.Tensor,
				keys: 	 torch.Tensor,
				values:  torch.Tensor,
				mask: torch.Tensor = None) -> torch.Tensor
		attention_weights = self.similarity(queries, keys)
		if self.normalize:
			attention_weights = masked_softmax(attention_weights, mask)
		attention = attention_weights.bmm(values)
		return attention

	def similarity(queries, keys):
		raise NotImplementedError 

def masked_softmax(matrix, mask):
	"""
	:param matrix
	:param mask
	
	"""

class DotProductAttention(Attention):
	@overrides
	def forward(self, 
				queries: torch.Tensor,
				keys: 	 torch.Tensor,
				values:  torch.Tensor) -> torch.Tensor
		return 

