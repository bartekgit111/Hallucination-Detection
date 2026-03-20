import numpy as np


def fit(model, data, epochs=3, print_update=10000):
    """
    This function trains Word2Vec model on dataset.

    parameters
    -model : word2vec
    -data :
    -epochs : number of training epochs.
    -print_update : frequency of printing average loss.

    procedure
    - shuffle token indexes each epoch
    - for each center word:
        - iterate over context window
        - generate negative samples
        - SGD update
    """
    for e in range(epochs):

        indexes = np.arange(len(data.text_as_indexes))
        np.random.shuffle(indexes)

        loss_sum = 0
        step = 0

        for i in indexes:

            center = data.text_as_indexes[i]

            for j in range(i - data.window_size, i + data.window_size + 1):

                if j == i or j < 0 or j >= len(data.text_as_indexes):
                    continue

                context = data.text_as_indexes[j]

                negatives = data.generate_negative_words(context)

                loss = model.train_step(center, context, negatives)

                loss_sum += loss
                step += 1

                if step % print_update == 0:
                    print(f"epoch {e+1}, step {step}, avg loss {loss_sum / print_update}")
                    loss_sum = 0

        print(f"-------end epoch {e+1}-------")
