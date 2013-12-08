cs224n-pa4
==========

Deep learning for NER

We reuse the CommandLineUtils.java in PA2 for simplify our parameter tuning process.

We do read and parse following parameter:

-window:  window size for the model, default is set to 13.

-layers:  hidden layers structure, default is set to 300, you can also set multi-level parameter, e.g.: 100, 80 this means two hidden layer neural network, the first one is 100 and the second one is 80.

-alpha:   learning rate, default is set to 0.001.

-regularize:  regularizing constanct, default is set to 0.0001.

-epoch:   number of iteration during the train, default it set to 10.

-data:    folder contains necessary train, dev and test data.

-train:   train filename.

-test:    test filename.

-dump:    dump the trained word vectors to filename if you specified, default is turned off.

-v:       print the objective function value after every checkpoint.


A example command with our best settings: window size = 13, only one hidden layer with size 250, default alpha and regulaization constant is like:


cd $PROJECT

java -Xmx2g -cp classes:extlib/ejml.jar cs224n.deep.NER -window 13 -layer 250 -epoch 10 -train train -test test -data $DATA
