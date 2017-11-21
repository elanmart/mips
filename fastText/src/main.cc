/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include <iostream>

#include "fasttext.h"
#include "args.h"

using namespace fasttext;

void printUsage() {
  std::cerr
    << "usage: fasttext <command> <args>\n\n"
    << "The commands supported by fasttext are:\n\n"
    << "  supervised              train a supervised classifier\n"
    << "  quantize                quantize a model to reduce the memory usage\n"
    << "  test                    evaluate a supervised classifier\n"
    << "  predict                 predict most likely labels\n"
    << "  predict-prob            predict most likely labels with probabilities\n"
    << "  to-fvecs                store hidden representations and output matrix for Fvecs benchmarking\n"
    << "  skipgram                train a skipgram model\n"
    << "  cbow                    train a cbow model\n"
    << "  print-word-vectors      print word vectors given a trained model\n"
    << "  print-sentence-vectors  print sentence vectors given a trained model\n"
    << "  nn                      query for nearest neighbors\n"
    << "  analogies               query for analogies\n"
    << std::endl;
}

void printQuantizeUsage() {
  std::cerr
    << "usage: fasttext quantize <args>"
    << std::endl;
}

void printTestUsage() {
  std::cerr
    << "usage: fasttext test <model> <test-data> [<k>]\n\n"
    << "  <model>      model filename\n"
    << "  <test-data>  test data filename (if -, read from stdin)\n"
    << "  <k>          (optional; 1 by default) predict top k labels\n"
    << std::endl;
}

void printPredictUsage() {
  std::cerr
    << "usage: fasttext predict[-prob] <model> <test-data> [<k>]\n\n"
    << "  <model>      model filename\n"
    << "  <test-data>  test data filename (if -, read from stdin)\n"
    << "  <k>          (optional; 1 by default) predict top k labels\n"
    << std::endl;
}

void printFvecsUsage() {
  std::cerr
    << "usage: fasttext to-fvecs <model> <test-data> <output>\n\n"
    << "  <model>      model filename\n"
    << "  <test-data>  test data filename (if -, read from stdin)\n"
    << "  <output>     where hidden.fvecs and wo.fvecs should be written\n"
    << std::endl;
}

void printPrintWordVectorsUsage() {
  std::cerr
    << "usage: fasttext print-word-vectors <model>\n\n"
    << "  <model>      model filename\n"
    << std::endl;
}

void printPrintSentenceVectorsUsage() {
  std::cerr
    << "usage: fasttext print-sentence-vectors <model>\n\n"
    << "  <model>      model filename\n"
    << std::endl;
}

void printPrintNgramsUsage() {
  std::cerr
    << "usage: fasttext print-ngrams <model> <word>\n\n"
    << "  <model>      model filename\n"
    << "  <word>       word to print\n"
    << std::endl;
}

void printTrainIndexUsage() {
  std::cerr
    << "usage: fasttext train-index <model> [args]\n\n"
    << "  <model>                       model filename\n\n"
    << "  Supported args are:\n\n"
    << "    index-size           [4096] index size passed to faiss\n"
    << "    index-quantizer      [Flat] index quantizer passed to faiss\n"
    << std::endl;
}

void printApproxPredictUsage() {
  std::cerr
    << "usage: fasttext approx-test <model> <test-data> [k] [nprobe]\n\n"
    << "  <model>                      model filename\n"
    << "  <test-data>                  test data filename\n"
    << "  <k>                  [5]     same as in fasttext.predict: we will output top k labels\n"
    << "  <nprobe>             [256]  `nprobe` search parameter passed to faiss\n"
    << std::endl;
}


void quantize(const std::vector<std::string>& args) {
  std::shared_ptr<Args> a = std::make_shared<Args>();
  if (args.size() < 3) {
    printQuantizeUsage();
    a->printHelp();
    exit(EXIT_FAILURE);
  }
  a->parseArgs(args);
  FastText fasttext;
  // parseArgs checks if a->output is given.
  fasttext.loadModel(a->output + ".bin");
  fasttext.quantize(a);
  fasttext.saveModel();
  exit(0);
}

void printNNUsage() {
  std::cout
    << "usage: fasttext nn <model> <k>\n\n"
    << "  <model>      model filename\n"
    << "  <k>          (optional; 10 by default) predict top k labels\n"
    << std::endl;
}

void printAnalogiesUsage() {
  std::cout
    << "usage: fasttext analogies <model> <k>\n\n"
    << "  <model>      model filename\n"
    << "  <k>          (optional; 10 by default) predict top k labels\n"
    << std::endl;
}

void test(const std::vector<std::string>& args) {
  if (args.size() < 4 || args.size() > 5) {
    printTestUsage();
    exit(EXIT_FAILURE);
  }
  int32_t k = 1;
  if (args.size() >= 5) {
    k = std::stoi(args[4]);
  }

  FastText fasttext;
  fasttext.loadModel(args[2]);

  std::string infile = args[3];
  if (infile == "-") {
    fasttext.test(std::cin, k);
  } else {
    std::ifstream ifs(infile);
    if (!ifs.is_open()) {
      std::cerr << "Test file cannot be opened!" << std::endl;
      exit(EXIT_FAILURE);
    }
    fasttext.test(ifs, k);
    ifs.close();
  }
  exit(0);
}

void predict(const std::vector<std::string>& args) {
  if (args.size() < 4 || args.size() > 5) {
    printPredictUsage();
    exit(EXIT_FAILURE);
  }
  int32_t k = 1;
  if (args.size() >= 5) {
    k = std::stoi(args[4]);
  }

  bool print_prob = args[1] == "predict-prob";
  FastText fasttext;
  fasttext.loadModel(std::string(args[2]));

  std::string infile(args[3]);
  if (infile == "-") {
    fasttext.predict(std::cin, k, print_prob);
  } else {
    std::ifstream ifs(infile);
    if (!ifs.is_open()) {
      std::cerr << "Input file cannot be opened!" << std::endl;
      exit(EXIT_FAILURE);
    }
    fasttext.predict(ifs, k, print_prob);
    ifs.close();
  }

  exit(0);
}

void trainIndex(const std::vector<std::string>& args) {
  auto fail = [](){ 
    printTrainIndexUsage() ; exit(EXIT_FAILURE); 
  };

  if ((args.size() < 3) || (args.size() > 5)) 
    fail();

  FastText fasttext;
  fasttext.loadModel(std::string(args[2]));

  if (fasttext.hasIndex()) {
    std::cerr << "Model is already trained. Exiting.\n\n";
    exit(EXIT_SUCCESS);
  }

  std::string index_size(args[3]);
  std::string index_quant(args[4]);

  fasttext.trainIndex(index_size, index_quant);
}

void approxPredict(const std::vector<std::string>& args) {

  auto fail = [](){ 
    printFvecsUsage() ; exit(EXIT_FAILURE); 
  };

  auto get_arg = [&args](int32_t idx, int32_t default_) {
    int32_t ret = default_;

    if (args.size() >= idx+1)
      ret = stoi(args[idx]);
    
    return ret;
  };

  if ((args.size() < 4) || (args.size() > 6))
    fail();

  FastText fasttext;
  fasttext.loadModel(std::string(args[2]));

  std::string fname(args[3]);
  std::ifstream ifs(fname);
  
  auto k      = get_arg(4, 1);
  auto nprobe = get_arg(5, 256);

  if (not ifs.is_open())
    fail();

  if (not fasttext.hasIndex())
    fail();

  fasttext.approxPredict(ifs, k, nprobe);

  exit(EXIT_SUCCESS); 
}

void toFvecs(const std::vector<std::string>& args) {
  if (args.size() != 5) {
    printFvecsUsage();
    exit(EXIT_FAILURE);
  }

  FastText fasttext;
  fasttext.loadModel(std::string(args[2]));

  std::string infile(args[3]);
  std::string outfile(args[4]);

  std::ifstream ifs(infile);
  std::ofstream ofs_h(outfile + ".hid.fvecs", std::ios::out | std::ios::binary);
  std::ofstream ofs_w(outfile + ".wo.fvecs",  std::ios::out | std::ios::binary);
  std::ofstream ofs_l(outfile + ".labels.txt");

  if (!ifs.is_open() || !ofs_h.is_open() || !ofs_w.is_open() || !ofs_l.is_open()) {
    std::cerr << "Some files could not be opened!" << std::endl;
    exit(EXIT_FAILURE);
  }

  fasttext.toFvecs(ifs, ofs_h, ofs_w, ofs_l);
  ifs.close();

  exit(0);
}

void printWordVectors(const std::vector<std::string> args) {
  if (args.size() != 3) {
    printPrintWordVectorsUsage();
    exit(EXIT_FAILURE);
  }
  FastText fasttext;
  fasttext.loadModel(std::string(args[2]));
  std::string word;
  Vector vec(fasttext.getDimension());
  while (std::cin >> word) {
    fasttext.getWordVector(vec, word);
    std::cout << word << " " << vec << std::endl;
  }
  exit(0);
}

void printSentenceVectors(const std::vector<std::string> args) {
  if (args.size() != 3) {
    printPrintSentenceVectorsUsage();
    exit(EXIT_FAILURE);
  }
  FastText fasttext;
  fasttext.loadModel(std::string(args[2]));
  Vector svec(fasttext.getDimension());
  while (std::cin.peek() != EOF) {
    fasttext.getSentenceVector(std::cin, svec);
    // Don't print sentence
    std::cout << svec << std::endl;
  }
  exit(0);
}

void printNgrams(const std::vector<std::string> args) {
  if (args.size() != 4) {
    printPrintNgramsUsage();
    exit(EXIT_FAILURE);
  }
  FastText fasttext;
  fasttext.loadModel(std::string(args[2]));
  fasttext.ngramVectors(std::string(args[3]));
  exit(0);
}

void nn(const std::vector<std::string> args) {
  int32_t k;
  if (args.size() == 3) {
    k = 10;
  } else if (args.size() == 4) {
    k = std::stoi(args[3]);
  } else {
    printNNUsage();
    exit(EXIT_FAILURE);
  }
  FastText fasttext;
  fasttext.loadModel(std::string(args[2]));
  fasttext.nn(k);
  exit(0);
}

void analogies(const std::vector<std::string> args) {
  int32_t k;
  if (args.size() == 3) {
    k = 10;
  } else if (args.size() == 4) {
    k = std::stoi(args[3]);
  } else {
    printAnalogiesUsage();
    exit(EXIT_FAILURE);
  }
  FastText fasttext;
  fasttext.loadModel(std::string(args[2]));
  fasttext.analogies(k);
  exit(0);
}

void train(const std::vector<std::string> args) {
  std::shared_ptr<Args> a = std::make_shared<Args>();
  a->parseArgs(args);
  FastText fasttext;
  fasttext.train(a);
  fasttext.saveModel();
  fasttext.saveVectors();
  if (a->saveOutput > 0) {
    fasttext.saveOutput();
  }
}

int main(int argc, char** argv) {
  std::vector<std::string> args(argv, argv + argc);
  if (args.size() < 2) {
    printUsage();
    exit(EXIT_FAILURE);
  }
  std::string command(args[1]);
  if (command == "skipgram" || command == "cbow" || command == "supervised") {
    train(args);
  } else if (command == "test") {
    test(args);
  } else if (command == "quantize") {
    quantize(args);
  } else if (command == "print-word-vectors") {
    printWordVectors(args);
  } else if (command == "print-sentence-vectors") {
    printSentenceVectors(args);
  } else if (command == "print-ngrams") {
    printNgrams(args);
  } else if (command == "nn") {
    nn(args);
  } else if (command == "analogies") {
    analogies(args);
  } else if (command == "predict" || command == "predict-prob" ) {
    predict(args);
  } else if (command == "to-fvecs") {
    toFvecs(args);
  } else {
    printUsage();
    exit(EXIT_FAILURE);
  }
  return 0;
}
