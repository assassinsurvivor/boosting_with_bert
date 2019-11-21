import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

print("Code is built using tensorflow:{}".format(tf.__version__))


#loading hyper-params
from config import *
params=config()
num_coders=params.num_coders         #number of encoders-decoders we want 
embed_dim=params.embed_dim           #dimensionality of word embedding
num_attn_head=params.num_attn_head   #total number of multi-head attention  

assert embed_dim%num_attn_head ==0

higher_dim=params.higher_dim
input_vocab_size=params.input_vocab_size
target_vocab_size=params.target_vocab_size
max_length_input=params.max_length_input
max_length_target=params.max_length_target

def scaled_dot_attention(k,q,v,mask):
    
    '''
    Params: key,query & value matrix along with padded_mask or a combination of padded_mask+look_ahead_mask(broadcasted)
    shape(k):  (batch_size,num_attn_head,seq_len_k,dim_k)
    shape(q):  (batch_size,num_attn_head,seq_len_k,dim_q)
    shape(v):  (batch_size,num_attn_head,seq_len_k,dim_v)
    
    assert dim_k==dim_q in order to do matrix multiplication
    
    shape(mask):(batch_size,1,1,seq_len)
    
    '''
    
    attn_logits=tf.matmul(q,k,transpose_b=True)  #shape :(batch_size,num_attn_head,seq_len_q,seq_len_k)
    attn_logits_scaled=tf.divide(attn_logits,tf.sqrt(tf.cast(tf.shape(k)[-1],tf.float32))) #dividing by the square root of embedding dimension to set variance of attn_logits to 1 and avoiding pushing the softmax scores towards 0 or 1 (hard softmax)
    
    if mask is not None:
        attn_logits_scaled+=(mask*-1e9) #making the mask values infinite small and pushing their logits scores towards zero
    
    attn_scaled=tf.nn.softmax(attn_logits_scaled,axis=-1) #softmax would be done across the keys dimension for getting the attention score of keys wrt to a query
    
    #shape(attn_scaled) :(batch_size,num_attn_head,seq_len_q,seq_len_k)
    
    out=tf.matmul(attn_scaled,v)
    
    return attn_scaled,out

class MultiHeadAttn(tf.keras.layers.Layer):
    
    def __init__(self,embed_dim,num_attn_head):
        super(MultiHeadAttn,self).__init__()
        
        '''
        
        One thing to note is that dimensionality % num of attention heads should be equal to zero because the idea is to 
        learn joint distribution from different linear projections of key,query and value matrix.
        
        params:
        dimension:the original dimension of initialized word embeddings
        attention_heads:how many different linear projections of key,query and value we want.
        
        '''
        
        self.embed_dim=embed_dim
        self.attn_head=num_attn_head
        
        assert self.embed_dim %self.attn_head ==0     #checking the above criteria
        
        self.depth =self.embed_dim//self.attn_head    #this would be the embedding dimension of k,q and v matrix after splitting in heads
        self.wk=tf.keras.layers.Dense(embed_dim)      #this layer will be used to linear project k matrix
        self.wq=tf.keras.layers.Dense(embed_dim)      #this layer will be used to linear project q matrix
        self.wv=tf.keras.layers.Dense(embed_dim)      #this layer will be used to linear project v matrix
        self.linear=tf.keras.layers.Dense(embed_dim)  #this layer will be used for linear projection after multi head split concatenation

    def __split__(self,batch_size,x):
        
        '''
        Use of this method is to split the incoming key,query,value matrix into a shape of {batch_size,self.attn_head,seq_length,self.depth} 
        This will help to learn the joint distribution of these matrices at different position wrt to different space.
        
        params:
        x:incoming k,q or v matrix
        batch_size:batch_size (number of queries)
        
        '''
        
        return tf.transpose(tf.reshape(x,(batch_size,-1,self.attn_head,self.depth)),perm=[0,2,1,3]) #the transpose is done to make shape of (batch_size,self.attn_head,seq_len,self.depth)

    
    def call(self,k,q,v,mask):
        
        batch_size=tf.shape(q)[0]
        
        #splitting the linearly transformed matrix to multiple small matrices.
        k=self.__split__(batch_size,self.wk(k))
        q=self.__split__(batch_size,self.wq(q))
        v=self.__split__(batch_size,self.wv(v))
        
        attn_scaled,out=scaled_dot_attention(k,q,v,mask) #shape(attn_scaled):(batch_size,num_attn_head,seq_len_q,seq_len_k)
                                                         #shape(out):(batch_size,num_attn_head,seq_len_q,self.depth)
            
        #restoring the output to shape:(batch_size,seq_len,self.embed_dim)
        
        out=tf.reshape(tf.transpose(out,perm=[0,2,1,3]),(batch_size,-1,self.embed_dim))
        
        #passing the output to the dense layer
        
        out=self.linear(out)
        
        return attn_scaled,out    

def ffn(higher_dim,embed_dim): #feed foward point wise network to pass the output obtained after multi head attn
    return tf.keras.Sequential([
                                tf.keras.layers.Dense(higher_dim,activation='relu'),
                                tf.keras.layers.Dense(embed_dim)
    ])
    
class encoder_layer(tf.keras.layers.Layer):
    
    def __init__(self,embed_dim,num_attn_head,higher_dim,drop_rate=0.05):
        super(encoder_layer,self).__init__()
        self.embed_dim=embed_dim
        self.attn_head=num_attn_head
        
        #initialising the multi head attention layer
        self.multi_attn=MultiHeadAttn(embed_dim,num_attn_head)
        
        #initialising the ffn network layer
        self.ffn=ffn(higher_dim,embed_dim)
        
        # dropout+layer norm
        self.dropout1=tf.keras.layers.Dropout(drop_rate)
        self.dropout2=tf.keras.layers.Dropout(drop_rate)
        
        '''
        LayerNormalization --> x=[x1,x2,x3,.......xn] where x is an embedding 
         
         1.calculate mean 
         2.calculate variance
         3.x1= alpha*(x1-mean)/sqrt(variance+epsilon)+beta where alpha and beta are learnable parameters.
         4.repeat this for all other xi.

        '''
        
        self.norm1=tf.keras.layers.LayerNormalization(epsilon=1e-9)
        self.norm2=tf.keras.layers.LayerNormalization(epsilon=1e-9)
        
        
    def call(self,x,mask,is_train):
        
        attn_scaled,out=self.multi_attn(x,x,x,mask) #out shape=(batch_size,seq_len,self.embed_dim)
        out=self.dropout1(out,training=is_train)    #out shape=(batch_size,seq_len,self.embed_dim)
        out=self.norm1(out+x)
        
        ffn_out=self.ffn(out)
        ffn_out=self.dropout2(ffn_out,training=is_train)
        ffn_out=self.norm2(ffn_out+out)
        
        return ffn_out                             #ffn_out shape=(batch_size,seq_len,self.embed_dim)

class decoder_layer(tf.keras.layers.Layer):
    
    def __init__(self,embed_dim,num_attn_head,higher_dim,drop_rate=0.05):
        super(decoder_layer,self).__init__()
        self.embed_dim=embed_dim
        self.attn_head=num_attn_head
        
        #initialising the multi head attention layer
        self.multi_attn1=MultiHeadAttn(embed_dim,num_attn_head) #this is decoder attention
        self.multi_attn2=MultiHeadAttn(embed_dim,num_attn_head) #this is encoder-decoder attention
        
        #initialising the ffn network layer
        self.ffn=ffn(higher_dim,embed_dim)
        
        # dropout+layer norm
        self.dropout1=tf.keras.layers.Dropout(drop_rate)
        self.dropout2=tf.keras.layers.Dropout(drop_rate)
        self.dropout3=tf.keras.layers.Dropout(drop_rate)

        
        self.norm1=tf.keras.layers.LayerNormalization(epsilon=1e-9)
        self.norm2=tf.keras.layers.LayerNormalization(epsilon=1e-9)
        self.norm3=tf.keras.layers.LayerNormalization(epsilon=1e-9)
        
    def call(self,x,encoder_out,look_ahead_mask,padded_mask,is_train):
        
        #this attn mechanism will take input of the target except the final word (example for hindi-english translation the input at encoder would be hindi matrix while here it would be english matrix )
        decoder_attn,decoder_out=self.multi_attn1(x,x,x,look_ahead_mask) #this will take broadcasted version of tf.math.maximum(look_ahead_mask+padded_mask)
        decoder_out=self.dropout1(decoder_out,training=is_train)
        decoder_out=self.norm1(x+decoder_out)
        
        #this attn mechanism will take the output of the previous english matrix attn as a query while output of encoder(hindi) would be its key and value
        encoder_decoder_attn,encoder_decoder_out=self.multi_attn2(encoder_out,decoder_out,encoder_out,padded_mask)#query here would be the output of decoder attention while key and value would be the output of encoder
        encoder_decoder_out=self.dropout2(encoder_decoder_out,training=is_train)
        encoder_decoder_out=self.norm2(encoder_decoder_out+decoder_out)
        
        #ffn network
        ffn_out=self.ffn(encoder_decoder_out)
        ffn_out=self.dropout3(ffn_out,training=is_train)
        ffn_out=self.norm3(ffn_out+encoder_decoder_out)
        
        
        return ffn_out,decoder_attn,encoder_decoder_attn


def angle_rate(pos,i,dim):
    return pos* 1/np.power(1e4,((2*(i/2))/dim))

def positional_embedding(max_sequence_length,embed_dim):
    
    max_sequence=np.arange(max_sequence_length).reshape(max_sequence_length,1)
    dimension=np.arange(embed_dim).reshape(1,embed_dim)
    
    embeddings=angle_rate(max_sequence,dimension,embed_dim)
    embeddings[:,0::2]=np.sin(embeddings[:,0::2])
    embeddings[:,1::2]=np.cos(embeddings[:,1::2]) #shape=(max_sequence_length,embedding_dim)
    
    return tf.cast(embeddings[np.newaxis,:,:],tf.float32) #shape=(batch_size,max_sequence_length,embedding_dim)


class encoder_nx(tf.keras.layers.Layer):
    
    def __init__(self,num_coders,num_attn_head,embed_dim,higher_dim,vocab_size,max_sequence_length,drop_rate=0.05):
        super(encoder_nx,self).__init__()
        self.num_encoders=num_coders
        self.embedding=tf.keras.layers.Embedding(vocab_size,embed_dim) #init embedding matrix of shape (vocab_Size,embed_dim)
        self.dropout1=tf.keras.layers.Dropout(drop_rate)
        self.get_position_embed=positional_embedding(max_sequence_length,embed_dim)
        self.encoders=[encoder_layer(embed_dim,num_attn_head,higher_dim,drop_rate) for i in range(self.num_encoders)]
        
    
    def call(self,x,mask,is_train):
        
        sequence_length=tf.shape(x)[1]
        x_embedding=self.embedding(x) #get embedding from embedding layer
        x_embedding+=self.get_position_embed[:,:sequence_length,:] #add position embedding
        x_embedding=self.dropout1(x_embedding,training=is_train)#dropout layer
        
        for i in self.encoders:
            x_embedding=i(x_embedding,mask,is_train)
            
        return x_embedding #this is the encoder output after n encoders       

class decoder_nx(tf.keras.layers.Layer):
    
    def __init__(self,num_coders,num_attn_head,embed_dim,higher_dim,vocab_size,max_sequence_length,drop_rate=0.05):
        super(decoder_nx,self).__init__()
        self.num_decoders=num_coders
        self.embedding=tf.keras.layers.Embedding(vocab_size,embed_dim) #init embedding matrix of shape (vocab_Size,embed_dim)
        self.dropout1=tf.keras.layers.Dropout(drop_rate)
        self.get_position_embed=positional_embedding(max_sequence_length,embed_dim)
        self.decoders=[decoder_layer(embed_dim,num_attn_head,higher_dim,drop_rate) for i in range(self.num_decoders)]
        
    def call(self,x,encoder_output,look_ahead_mask,padded_mask,is_train):
        sequence_length=tf.shape(x)[1]
        x_embedding=self.embedding(x) #get embedding from embedding layer
        x_embedding+=self.get_position_embed[:,:sequence_length,:] #add position embedding
        x_embedding=self.dropout1(x_embedding,training=is_train)#dropout layer
        
        attn_dict=dict()
        
        for idx,i in enumerate(self.decoders):
            x_embedding,decoder_attn,enc_dec_attn=i(x_embedding,encoder_output,look_ahead_mask,padded_mask,is_train)
            attn_dict["decoder attention for n:{}".format(idx+1)]=decoder_attn
            attn_dict["encoder decoder attention for n:{}".format(idx+1)]=enc_dec_attn
            
        return x_embedding,attn_dict
            
class Transformer(tf.keras.Model):
    def __init__(self,num_coders,num_attn_head,embed_dim,higher_dim,input_vocab_size,target_vocab_size,max_length_input,max_length_target,drop_rate=0.05,):
        super(Transformer,self).__init__()
        self.encoder=encoder_nx(num_coders,num_attn_head,embed_dim,higher_dim,input_vocab_size,max_length_input,drop_rate)
        self.decoder=decoder_nx(num_coders,num_attn_head,embed_dim,higher_dim,target_vocab_size,max_length_target,drop_rate)
        self.final_layer=tf.keras.layers.Dense(target_vocab_size)
        
    def call(self,inp,tar,padded_mask,look_ahead_mask,is_train):
        encoder_output=self.encoder(inp,padded_mask,is_train)
        decoder_output,attn_weight_dict=self.decoder(tar,encoder_output,look_ahead_mask,padded_mask,is_train)
        final_output=self.final_layer(decoder_output)
        return final_output,attn_weight_dict
        
        
class LearningRate(tf.keras.optimizers.schedules.LearningRateSchedule):
    
    def __init__(self,embed_dim=600,warmup_steps=4000):
        super(LearningRate,self).__init__()
        self.dim=tf.cast(embed_dim,tf.float32)
        self.warmup_steps=warmup_steps
        
    def __call__(self,step):
        
        return tf.math.rsqrt(self.dim) * tf.math.minimum(tf.math.rsqrt(step),step*(self.warmup_steps**-1.5))

lr=LearningRate(embed_dim,warmup_steps=4000)
optimizer=tf.keras.optimizers.Adam(lr,epsilon=1e-9)

def calculate_loss(real,pred):
    mask=tf.math.logical_not(tf.math.equal(real,0)) #finding the padded values
    loss_calc=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,reduction='none')
    loss=loss_calc(real,pred)
    mask=tf.cast(mask,dtype=loss.dtype)
    loss=tf.reduce_mean(loss*mask)
    return loss
    
def loss_acc_metric():
    return tf.keras.metrics.Mean(name='train_loss'),tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

train_loss,train_accuracy=loss_acc_metric()

def create_padded_mask(seq):
    #this function is used to create mask matrix for padded tokens
    masked=tf.cast(tf.math.equal(seq,0),tf.float32) #(batch_Size,seq_length)
    masked=masked[:,tf.newaxis,tf.newaxis,:] #(batch_size,1,1,seq_length) so as to add to attn_logits of shape:(batch_size,num_attn_head,seq_len_q,seq_len_k)
    return masked

def create_look_ahead_mask(seq_length):
    #this function is used at decoder size to mask the future tokens
    
    #example:for input:Je susis estudiant and output: I am a student for prediction 'I' at decoder everything should be masked\
    #while predicting 'am' except 'I' all other should be masked and so on.
    
    #Notice the shape of this mask is (seq_length,seq_length) which can not be added directly to attn_logits.
    #Why this is done is explained in the below function
    return 1- tf.linalg.band_part(tf.ones((seq_length,seq_length)),-1,0) 


def create_mask(inp,tar):
    #shape inp &tar:(batch_size,seq_length)
    encoder_mask=create_padded_mask(inp)
    decoder_mask=create_padded_mask(tar)
    decoder_look_ahead=create_look_ahead_mask(tf.shape(tar)[1])
    
    #This is done because if there exists a padded value ,it doesn't matter to look into its value or not.\
    #Also doing the comparison of two matrics of shape (batch_size,1,1,seq_length) with (seq_length,seq_length)\
    #would result in broadcasting and generate output matrix of shape :(batch_size,1,seq_length,seq_length) which we want.
    
    combined_mask=tf.math.maximum(decoder_mask,decoder_look_ahead)
    return encoder_mask,combined_mask


transformer=Transformer(num_coders=num_coders,num_attn_head=num_attn_head,embed_dim=embed_dim,higher_dim=higher_dim,\
                        input_vocab_size=input_vocab_size,target_vocab_size=target_vocab_size\
                        ,max_length_input=max_length_input,max_length_target=max_length_target,drop_rate=0.05)


#creating graph
inp_signature=[tf.TensorSpec(shape=(None,None),dtype=tf.int64),tf.TensorSpec(shape=(None,None),dtype=tf.int64),tf.TensorSpec(shape=(None),dtype=tf.bool)]
@tf.function(input_signature=inp_signature)
def train(inp,target,flag):
    decoder_input=target[:,:-1] #example for output '<start> I am a student <eos>' the values that should be fed to the decoder should be '<start> I am a'
    decoder_output=target[:,1:] # 'I am a student <eos>'
    encoder_mask,decoder_look_ahead=create_mask(inp,decoder_input)
    with tf.GradientTape() as tape:
        output,attn_dict=transformer(inp,decoder_input,encoder_mask,decoder_look_ahead,True)
        if flag==True:
            print(transformer.summary())
        loss_=calculate_loss(decoder_output,output)
    
    gradients=tape.gradient(loss_,transformer.trainable_variables)
    optimizer.apply_gradients(zip(gradients,transformer.trainable_variables))
    
    train_loss.update_state(loss_)
    train_accuracy.update_state(decoder_output,output)


