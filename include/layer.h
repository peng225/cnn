#include <vector>

typedef std::pair<int, int> DataSize;

const float GAMMA = 0.1;

class ConvolutionLayerTest;

class Layer
{
public:
    virtual ~Layer(){};
    void setInputInfo(const DataSize& size, int numInputChannel);
    virtual void calcOutputSize() = 0;
    DataSize getOutputSize() const{return outputSize;}
    int getNumOutputChannel() const{return numOutputChannel;}
    virtual std::vector<float> apply(const std::vector<float>& input) const = 0;
    virtual void initWeight(){};
    virtual std::vector<float> updateWeight(const std::vector<float>& input,
                const std::vector<float>& output,
                const std::vector<float>& propError) = 0;

protected:
    DataSize inputSize;
    DataSize outputSize;
    int numInputChannel;
    int numOutputChannel;

};

class ConvolutionLayer : public Layer
{
public:
    ConvolutionLayer(int zeroPad, int windowSize, int numOutputChannel);

    void calcOutputSize() override;
    std::vector<float> apply(const std::vector<float>& input) const override;
    void initWeight();
    std::vector<float> updateWeight(const std::vector<float>& input,
                const std::vector<float>& output,
                const std::vector<float>& propError);

private:
    std::vector<float> weight;
    std::vector<float> bias;
    int zeroPad;
    int windowSize;
};

class ReLULayer : public Layer
{
public:
    void calcOutputSize() override;
    std::vector<float> apply(const std::vector<float>& input) const override;
    std::vector<float> updateWeight(const std::vector<float>& input,
                const std::vector<float>& output,
                const std::vector<float>& propError);

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
                const std::vector<float>& propError);


private:
    int zeroPad;
    int windowSize;
};

class FullConnectLayer : public Layer
{
public:
    FullConnectLayer(DataSize size);

    void calcOutputSize() override;
    std::vector<float> apply(const std::vector<float>& input) const override;
    void initWeight();
    std::vector<float> updateWeight(const std::vector<float>& input,
                const std::vector<float>& output,
                const std::vector<float>& propError);

private:
    std::vector<float> weight;
    float bias;
};


