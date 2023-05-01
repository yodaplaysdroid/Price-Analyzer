import data_processing
import model_training

if __name__ == "__main__":
    
    xau = data_processing.Stock("data/XAUUSD.csv")
    xau.get_train_test()

    rnn = model_training.RNN()
    rnn.auto_train()