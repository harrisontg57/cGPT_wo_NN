# cGPT_wo_NN
Text Generation with a graph rather than neural networks
REQUIREMENTS: python nltk package
Usage:
RUN docGraphUI.py as
python3 docGraphUI.py training_document.txt -h<N> -<mode>
Where <N> is the depth h to train sequences and <mode> is the generation mode (currently rand or max)

NEXT you will be prompted for the generation text which can be either a .txt file or a sentence in double quotes "".
THEN you will be asked for the number of new words to generate followed by the output with these newly generated words.

THEN you will be repeatedly asked "Add Word(s) in Quotes followed by N to generate N new words (if N > 1):"

EXIT with ctrl+C

Words not seen in the training document currently throw errors.

The file simplewiki_dogs.txt contains the text from the simple english wikipedia page for 'dogs' as a test training set.

SUGGESTED TEST:
docGraphUI.py simplewiki_dogs.txt -h50 -rand
