#include <iostream>
#include <string>
#include <fstream>
#include <algorithm>
#include <unordered_set>
#include <unordered_map>
#include <stdio.h>
#include <sys/types.h>
#include <dirent.h>
#include <sstream>
#include <vector>
#include <cmath>
#include <float.h>
#include <limits.h>

using namespace std;
#define g(x) (1.0/(1.0+exp(-x)))
#define gprime(x) (g(x)*(1-g(x))) 

// Print out progress bar
#define PBSTR "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
#define PBWIDTH 60

void printProgress(double percentage, string info) {
    int val = (int) (percentage * 100);
    int lpad = (int) (percentage * PBWIDTH);
    int rpad = PBWIDTH - lpad;
    printf("%s \r%3d%% [%.*s%*s] ", info.c_str(), val, lpad, PBSTR, rpad, "");
    fflush(stdout);
}

// Process words to match words in the vocabulary file
string ProcessWord(string s)
{
    string t;
    // Remove punctuation.
    for (int i = 0; i < s.size(); i++)
        if (!ispunct(s[i]))
            t += s[i];

    // Convert to lower case.
    std::transform(t.begin(), t.end(), t.begin(), ::tolower);
    
    return t;

}

// The feature we use is the number of occurances of a word in the file
vector<double> GetFeature(string filename, unordered_map<string, int> &word_to_idx)
{
    /*
    Args:
    - filename: path name to the data file
    - word_to_idx: a mapping where a word maps to a preceptron weight's index.
    */

    // Open file.
    ifstream in;
    in.open(filename.c_str());
    if (!in.is_open())
    {
        cerr << "File not found: " << filename << endl;
        exit(1);
    }
    
    //obtain feature
    vector<double> feature; // initialize an empty vector of size of word list. 
    for(unsigned int i=0; i<word_to_idx.size(); i++){
        feature.push_back(0);
    }
    string line;
    while(getline(in, line)){//in case there are multiple lines of data
        stringstream ss(line);
        string word;
        while (ss >> word){
            word = ProcessWord(word);
            if(word_to_idx.find(word) != word_to_idx.end()){
                feature.at(word_to_idx.find(word)->second)++;//increments per instance of a word
            }
        }
    }
    in.close();
    
    return feature;
}

void save_trained_weights(vector<double> &weight)
{
    ofstream output_file("trained_weights");
    ostream_iterator<double> output_iterator(output_file, "\n");
    copy(weight.begin(), weight.end(), output_iterator);
    output_file.close();
}

int main()
{       
    // Open vocabulary file
    ifstream dict_file;
    dict_file.open("imdb.vocab");
    if (!dict_file.is_open())
    {
        cerr << "Dictionary not found." << endl;
        exit(1);
    }

    //Create word to weight index map dictionary based on the imdb.vocab
    unordered_map<string, int> wordToIdx;
    string line;
    int idx = 0;
    while(getline(dict_file, line)){
        line = ProcessWord(line);
        if(wordToIdx.find(line) == wordToIdx.end()){
            wordToIdx.insert(make_pair(line, idx));//should error check for duplicates
            idx++; 
        }
        // wordToIdx.insert(make_pair(line, idx));
        // idx++;
    }

    dict_file.close();

    // initialize accuracy output file
    ofstream output_acc_file;
    output_acc_file.open("accuracy.txt");

    //Initialize perceptron weights and training parameters. 
    vector<double> weights;
    for(auto i : wordToIdx){
        weights.push_back(-0.03);
    }

    float alpha = 1; 
    int total_epoch = 20;

    // Train the weights
    for (int epoch = 0; epoch < total_epoch; ++epoch) 
    {   
        // Read in train_list
        string file_name;
        string delimiter = "\t";
        ifstream train_file;
        train_file.open("training_list");
        float line_count = 0.0;
        cout << "Epoch " << epoch;
        // Update perceptron weights based on each data instance
        while (getline(train_file, file_name))
        {   
            stringstream ss(file_name); // should only be those two things...
            string classify;
            ss >> file_name;
            ss >> classify;
            double weightedSum = 0.0;
            vector<double> singleEntry = GetFeature(file_name, wordToIdx);
            for(auto i : wordToIdx){
                weightedSum += (weights.at(i.second) * singleEntry.at(i.second));   
            }
            
            for(auto i : wordToIdx){
                weights.at(i.second) -= (alpha * (g(weightedSum) - stod(classify)) * (gprime(weightedSum) * singleEntry.at(i.second)));
            }

            // Output current progress on training with progress bar
            line_count += 1;
            printProgress(line_count/10000, "Epoch "+ to_string(epoch));
        } 
        train_file.close();
        cout << endl;

        // Save trained weights
        save_trained_weights(weights);
        
        // Evaluate trained weights with test data
        ifstream test_file;
        test_file.open("test_list");
        float correct_pred = 0.0; // should update during evaluation
        line_count = 0.0;

        //Predict negative or positive review with current trained preceptron weights
        while (getline(test_file, file_name))
        {
            stringstream ss(file_name); // should only be those two things...
            string classify;
            ss >> file_name;
            ss >> classify;
            double weightedSum = 0.0;
            vector<double> singleEntry = GetFeature(file_name, wordToIdx);
            for(auto i : wordToIdx){
                weightedSum += (singleEntry.at(i.second) * weights.at(i.second));
            }


            //Check if the prediction is correct. If so, increase correct_pred by one.
            if(classify == "0" && g(weightedSum) <= 0.5){
                correct_pred++;
            }
            else if(classify == "1" && g(weightedSum) > 0.5){
                correct_pred++;
            }
            // Output current progress on evaluation with progress bar
            line_count += 1.0;
            printProgress(line_count/1000, "Evaluating...");
        } 
        test_file.close();

        // Compute prediction accuracy
        float acc = (correct_pred/line_count)*100.0;
        printf("Accuracy: %.2f \n", acc);
        output_acc_file << "Epoch " << epoch << ": Accuracy = " << acc << "%" << endl;
    } 
    output_acc_file.close();
}
