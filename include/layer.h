#pragma once
#include <vector>
#include <fstream>
#include <mutex>

typedef std::pair<int, int> DataSize;

const float GAMMA = 0.02;  // 学習率
const float LAMBDA = 0.001;  // L2正則化の係数

class ConvolutionLayerTest;

class Layer
{
public:
    Layer();
    virtual ~Layer(){};
    void setInputInfo(const DataSize& size, int numInputChannel);
    virtual void calcOutputSize() = 0;
    DataSize getOutputSize() const{return outputSize;}
    int getNumOutputChannel() const{return numOutputChannel;}
    virtual std::vector<float> apply(const std::vector<float>& input) const = 0;
    virtual void initWeight(){};
    virtual std::vector<float> updateWeight(const std::vector<float>& input,
                const std::vector<float>& output,
                const std::vector<float>& propError,
                double reduceRate) = 0;
    virtual void saveWeight(std::ofstream& ofs) const{};
    virtual void loadWeight(std::ifstream& ifs){};
    void setVerboseMode(bool mode){verbose = mode;};
    virtual void flush(){};

protected:
    DataSize inputSize;
    DataSize outputSize;
    int numInputChannel;
    int numOutputChannel;
    bool verbose;
};

class ConvolutionLayer : public Layer
{
friend class ConvolutionLayerTest;
public:
    ConvolutionLayer(int zeroPad, int windowSize, int numOutputChannel);

    void calcOutputSize() override;
    std::vector<float> apply(const std::vector<float>& input) const override;
    void initWeight() override;
    std::vector<float> updateWeight(const std::vector<float>& input,
                const std::vector<float>& output,
                const std::vector<float>& propError,
                double reduceRate = 1.0) override;
    void dumpWeight() const;
    void saveWeight(std::ofstream& ofs) const override;
    void loadWeight(std::ifstream& ifs) override;
    void flush() override;

private:
    std::vector<float> weight;
    std::vector<float> bias;
    std::vector<float> diffWeight;
    std::vector<float> diffBias;
    int zeroPad;
    int windowSize;
    // weight, bias両方のロックを取る場合、
    // weight -> biasの順に取ること
    std::mutex mtxDiffWeight;
    std::mutex mtxDiffBias;
};

class ReLULayer : public Layer
{
public:
    void calcOutputSize() override;
    std::vector<float> apply(const std::vector<float>& input) const override;
    std::vector<float> updateWeight(const std::vector<float>& input,
                const std::vector<float>& output,
                const std::vector<float>& propError,
                double reduceRate = 1.0) override;

private:

};

class PoolingLayer : public Layer
{
public:
    PoolingLayer(int zeroPad, int windowSize) :
        zeroPad(zeroPad), windowSize(windowSize){}

    void calcOutputSize() override;
    std::vector<float> apply(const std::vector<float>& input) const override;
    std::vector<float> updateWeight(const std::vector<float>& input,
                const std::vector<float>& output,
                const std::vector<float>& propError,
                double reduceRate = 1.0) override;


private:
    int zeroPad;
    int windowSize;
};

class FullConnectLayer : public Layer
{
friend class FullConnectLayerTest;
public:
    FullConnectLayer(DataSize size);

    void calcOutputSize() override;
    std::vector<float> apply(const std::vector<float>& input) const override;
    void initWeight() override;
    std::vector<float> updateWeight(const std::vector<float>& input,
                const std::vector<float>& output,
                const std::vector<float>& propError,
                double reduceRate = 1.0) override;
    void saveWeight(std::ofstream& ofs) const override;
    void loadWeight(std::ifstream& ifs) override;
    void flush() override;

private:
    std::vector<float> weight;
    float bias;
    std::vector<float> diffWeight;
    float diffBias;
    // weight, bias両方のロックを取る場合、
    // weight -> biasの順に取ること
    std::mutex mtxDiffWeight;
    std::mutex mtxDiffBias;
};

class SoftmaxLayer : public Layer
{
public:
    void calcOutputSize() override;
    std::vector<float> apply(const std::vector<float>& input) const override;
    std::vector<float> updateWeight(const std::vector<float>& input,
                const std::vector<float>& output,
                const std::vector<float>& propError,
                double reduceRate = 1.0) override;

private:
    std::vector<float> softmax(const std::vector<float>& input) const;

};

class SigmoidLayer : public Layer
{
public:
    void calcOutputSize() override;
    std::vector<float> apply(const std::vector<float>& input) const override;
    std::vector<float> updateWeight(const std::vector<float>& input,
                const std::vector<float>& output,
                const std::vector<float>& propError,
                double reduceRate = 1.0) override;

private:
    std::vector<float> sigmoid(const std::vector<float>& input) const;

};

class StandardizeLayer : public Layer
{
public:
    StandardizeLayer(int nb);
    void calcOutputSize() override;
    std::vector<float> apply(const std::vector<float>& input) const override;
    std::vector<float> updateWeight(const std::vector<float>& input,
                const std::vector<float>& output,
                const std::vector<float>& propError,
                double reduceRate = 1.0) override;
    void standardize(std::vector<float>::iterator leftItr,
        std::vector<float>::iterator rightItr) const;
    float getMean(std::vector<float>::const_iterator leftItr,
        std::vector<float>::const_iterator rightItr) const;
    float getStddev(std::vector<float>::const_iterator leftItr,
        std::vector<float>::const_iterator rightItr, float mean) const;
private:
    int numBatch;
};

