import numpy as np
import matplotlib.pyplot as plt
import gensim
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from gensim.models.keyedvectors import KeyedVectors

# Basic info about the data
def data_info(df, selected_data, abstracts, headings):
    print("\n------------------------------------------------------")
    tobeprocessed = selected_data.shape[0]
    total = df.shape[0]
    percentage = tobeprocessed / total     
    print("\nROWS and COLUMNS of the start data:", df.shape)
    print("Size of the data to be processed : ", tobeprocessed)
    #print("Length of the Abstract Array: ", len(abstracts))
    #print("Length of the Heading Array: ", len(headings))
    print ("Percentage of data that will be processed: ", "{:.0%}".format(percentage))
    print("--------------------------------------------------------\n")


def encode_abstract(abstracts, tokenizer, MAX_LENGTH):
    tokenizer.fit_on_texts(abstracts)
    abstracts_sequences = tokenizer.texts_to_sequences(abstracts)
    abstracts_padded = pad_sequences(abstracts_sequences, maxlen=MAX_LENGTH)
    word_index = tokenizer.word_index
    return abstracts_padded, word_index


def encode_label(headings, label_encoder):
    # convert string labels to integers
    headings_encoded = label_encoder.fit_transform(headings)
    return headings_encoded


def get_word2vec_embedding(word2vec_file, tokenizer, NUMBER_OF_WORDS):    
    # Load pre-trained Word2Vec model
    print("\nLoad word2vec vector ")
    word2vec = gensim.models.KeyedVectors.load_word2vec_format(word2vec_file, binary=True)
    
    # Create an embedding matrix
    embedding_dim = word2vec.vector_size    
    embedding_matrix = np.zeros((NUMBER_OF_WORDS, embedding_dim))
    
    for word, i in tokenizer.word_index.items():
        if i < NUMBER_OF_WORDS:
            try:
                embedding_vector = word2vec[word]
                embedding_matrix[i] = embedding_vector
            except KeyError:
                # Words not found in Word2Vec will be all zeros
                pass
    return embedding_dim, embedding_matrix


def predict_this (X_test, y_test, label_encoder, model):
    # Predicting on the test set
    predictions = model.predict(X_test)
    predicted_labels = label_encoder.inverse_transform(np.argmax(predictions, axis=1))
    actual_labels = label_encoder.inverse_transform(np.argmax(y_test, axis=1))

    # Display results
    print("\n")
    found= 0
    for i, text in enumerate(X_test):
        if (actual_labels[i] == predicted_labels[i]):
            found += 1
            #print(f"Text: {' '.join([list(word_index.keys())[list(word_index.values()).index(idx)] for idx in text if idx != 0])}")
            #print(f"Actual Label: {actual_labels[i]}")
            #print(f"Predicted Label: {predicted_labels[i]}")
            #print()
    print("\nNumber of words predicted correctly is: ", found)


def plot_this(what_to_plot, title, legend,  xlabel, ylabel, history):
    # Plot accuracy
    plt.plot(history.history[what_to_plot], label = legend)
    plt.title(title)
    plt.xlabel(xlabel,)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()


def get_df_subset(selected_data, SUBSET):    
    # get the records with a subject heading count > n
    filtered_sh = selected_data.groupby('Subject Headings').filter(lambda x: len(x) >= SUBSET)    
    distinct_sh = filtered_sh['Subject Headings'].unique()  
    counts = filtered_sh ['Subject Headings'].value_counts()     
    return filtered_sh  
