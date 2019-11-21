class config():

    def __init__(self):

        #Hyper-params
        self.num_coders=2          #number of encoders-decoders we want 
        self.embed_dim=300          #dimensionality of word embedding
        self.num_attn_head=4       #total number of multi-head attention  
        self.higher_dim=600
        self.input_vocab_size=4029
        self.target_vocab_size=3994
        self.max_length_input=4029
        self.max_length_target=3994

