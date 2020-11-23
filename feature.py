'''
>>> tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
>>> model = BertModel.from_pretrained('bert-base-uncased', return_dict=True)

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
>>> outputs = model(**inputs)

>>> last_hidden_states = outputs.last_hidden_state

forward(input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, encoder_hidden_states=None, encoder_attention_mask=None, output_attentions=None, output_hidden_states=None, return_dict=None)[SOURCE]
The BertModel forward method, overrides the __call__() special method.

Parameters
input_ids (torch.LongTensor of shape (batch_size, sequence_length)) –
Indices of input sequence tokens in the vocabulary.

attention_mask (torch.FloatTensor of shape (batch_size, sequence_length), optional)
Mask to avoid performing attention on padding token indices. Mask values selected in [0, 1]:
1 for tokens that are not masked

token_type_ids (torch.LongTensor of shape (batch_size, sequence_length), optional) –
Segment token indices to indicate first and second portions of the inputs. Indices are selected in [0, 1]:
0 corresponds to a sentence A token,
1 corresponds to a sentence B token.

What are token type IDs?

position_ids (torch.LongTensor of shape (batch_size, sequence_length), optional) –

Indices of positions of each input sequence tokens in the position embeddings. Selected in the range [0, config.max_position_embeddings - 1].

What are position IDs?

head_mask (torch.FloatTensor of shape (num_heads,) or (num_layers, num_heads), optional) –
------------------------------------------------------------------------------------------------
Mask to nullify selected heads of the self-attention modules. Mask values selected in [0, 1]:
1 indicates the head is not masked,
0 indicates the head is masked.
------------------------------------------------------------------------------------------------

>>> from transformers import BertTokenizer
>>> tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
>>> sequence_a = "HuggingFace is based in NYC"
>>> sequence_b = "Where is HuggingFace based?"

>>> encoded_dict = tokenizer(sequence_a, sequence_b)
'''