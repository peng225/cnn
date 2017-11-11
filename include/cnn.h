#include <cstdint>
#include <list>
#include <vector>
#include <utility>
#include <memory>

typedef std::pair<int, int> DataSize;

class Layer
{
public:
    virtual ~Layer(){};
    void setInputSize(const DataSize& size){inputSize = size;}
    virtual void calcOutputSize() = 0;
    DataSize getOutputSize() const{return outputSize;}
    virtual std::vector<float> apply(const std::vector<float>& input) const = 0;
    virtual void initWeight(){};

protected:
    DataSize inputSize;
    DataSize outputSize;

};

class ConvolutionLayer : public Layer
{
public:
    ConvolutionLayer(int zeroPad, int windowSize); 

    void calcOutputSize() override;
    std::vector<float> apply(const std::vector<float>& input) const override;

private:
    std::vector<float> weight;
    int zeroPad;
    int windowSize;
};

class ReLULayer : public Layer
{
public:
    void calcOutputSize() override;
    std::vector<float> apply(const std::vector<float>& input) const override;

private:

};

class PoolingLayer : public Layer
{
public:
    PoolingLayer(int zeroPad, int windowSize) :
        zeroPad(zeroPad), windowSize(windowSize){}

    void calcOutputSize() override;
    std::vector<float> apply(const std::vector<float>& input) const override;

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

private:
    std::vector<float> weight;

};

class DeepNetwork
{
public:
    bool setInputInfo(DataSize size);
    void addLayer(std::shared_ptr<Layer> layer);
    std::vector<float> feedInput(const std::vector<float>& input) const;
private:
    DataSize inputSize;
    std::list<std::shared_ptr<Layer>> layers;
};
