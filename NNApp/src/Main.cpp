#include "pch.h"

#include <fstream>
#include <string>
#include <sstream>
#include <Awincs.h>
#include "NeuralNetwork.h"

using InputRef = std::shared_ptr<Awincs::InputComponent>;
using ButtonRef = std::shared_ptr<Awincs::ButtonComponent>;
using PanelRef = std::shared_ptr<Awincs::PanelComponent>;
using ComponentRef = std::shared_ptr<Awincs::Component>;
using WindowRef = std::shared_ptr<Awincs::WindowController>;
using TitleBarRef = std::shared_ptr<Awincs::TitleBarComponent>;

using namespace std::literals::string_literals;

struct
{
    PanelRef statusBar              = nullptr;
    InputRef layers                 = nullptr;
    InputRef classify               = nullptr;
    PanelRef classifyOutput         = nullptr;
    InputRef loadNeuralNetwork      = nullptr;
    InputRef saveNeuralNetwork      = nullptr;
    InputRef loadTrainingData       = nullptr;
} inputs;

struct
{

    TitleBarRef titleBar            = nullptr;
    WindowRef window                = nullptr;
    ComponentRef mainNNPanel        = nullptr;
    ComponentRef loadNNPanel        = nullptr;
    ComponentRef saveNNPanel        = nullptr;
    ComponentRef trainNNPanel       = nullptr;
} panels;

const wchar_t* MAIN_NN_PANEL_TITLE_BAR = L"Nural Network";
const wchar_t* SAVE_NN_PANEL_TITLE_BAR = L"Nural Network - Save";
const wchar_t* LOAD_NN_PANEL_TITLE_BAR = L"Nueral Network - Load";
const wchar_t* TRAIN_NN_PANEL_TITLE_BAR = L"Nueral Network - Train";

NN::NeuralNetwork nn;


/*********************************************************/
/*                      UIParse                          */
void setupNeuralNetwork(std::vector<int> newLayers, bool force = false)
{
    static std::vector<int> layers = {};

    if (force || newLayers != layers)
    {
        layers = newLayers;
        nn.clear();

        for (const auto& layer : layers)
            nn.pushLayer(layer);

        const double maxLim = 1;
        const double minLim = -1;

        for (size_t i = 0; i < layers.size() - 1; i++)
            nn.setupWeights(i, i + 1, NN::NeuralNetwork::randomizeWeights(minLim, maxLim, layers[i], layers[i + 1]));
    }
}

/*********************************************************/
/*********************************************************/

/*********************************************************/
/*                      UIParse                          */

template<typename T>
std::vector<T> parseVectorFromString(std::wstring str, wchar_t delimiter = L',')
{
    static_assert(false);
}

template<>
std::vector<double> parseVectorFromString(std::wstring str, wchar_t delimiter)
{
    expect(str.size() > 0);

    std::vector<double> output;

    auto delim = str.begin();
    while (delim != str.end())
    {
        auto nextDelim = std::find(delim, str.end(), delimiter);
        output.emplace_back(std::stod(std::wstring(delim, nextDelim)));
        if (nextDelim == str.end())
            return output;
        delim = nextDelim + 1;
    }

    return output;
}

template<>
std::vector<int> parseVectorFromString<int>(std::wstring str, wchar_t delimiter)
{
    expect(str.size() > 0);

    std::vector<int> output;

    auto delim = str.begin();
    while (delim != str.end())
    {
        auto nextDelim = std::find(delim, str.end(), delimiter);
        output.emplace_back(std::stoi(std::wstring(delim, nextDelim)));
        if (nextDelim == str.end())
            return output;
        delim = nextDelim + 1;
    }

    return output;
}
/*********************************************************/
/*********************************************************/

/*********************************************************/
/*                  Panel switch handlers                */
auto onLoadNNClick = [](const Awincs::Component::Point& p)
{
    if (panels.loadNNPanel)
        panels.loadNNPanel->setParent(panels.window);

    panels.mainNNPanel->unsetParent();
    
    if (panels.titleBar)
        panels.titleBar->setText(LOAD_NN_PANEL_TITLE_BAR);
};

auto onSaveNNClick = [](const Awincs::Component::Point& p)
{
    if (panels.saveNNPanel)
        panels.saveNNPanel->setParent(panels.window);

    panels.mainNNPanel->unsetParent();

    if (panels.titleBar)
        panels.titleBar->setText(SAVE_NN_PANEL_TITLE_BAR);
};

auto onTrainNNClick = [](const Awincs::Component::Point& p)
{
    if (panels.trainNNPanel)
        panels.trainNNPanel->setParent(panels.window);

    panels.mainNNPanel->unsetParent();

    if (panels.titleBar)
        panels.titleBar->setText(TRAIN_NN_PANEL_TITLE_BAR);
};

auto onBackClick = [](const Awincs::Component::Point& p)
{
    if (panels.loadNNPanel)
        panels.loadNNPanel->unsetParent();
    if (panels.saveNNPanel)
        panels.saveNNPanel->unsetParent();
    if (panels.trainNNPanel)
        panels.trainNNPanel->unsetParent();

    panels.mainNNPanel->setParent(panels.window);

    if (panels.titleBar)
        panels.titleBar->setText(MAIN_NN_PANEL_TITLE_BAR);
};

/*********************************************************/
/*********************************************************/


/*********************************************************/
/*              Particular Panel Load & Save             */

auto onLoadNNDataClick = [](const Awincs::Component::Point& p)
{
    std::wstring filename = inputs.loadNeuralNetwork->getText();
    std::ifstream ifs(filename, std::ios_base::binary);

    if (!ifs)
    {
        MessageBox(NULL, L"Failed to open neural network file", L"Neural network open failed!", MB_OK | MB_ICONWARNING);
        return;
    }

    nn.clear();

    size_t lCount = 0;
    ifs.read(reinterpret_cast<char*>(&lCount), sizeof(lCount));

    for (size_t i = 0; i < lCount; i++)
    {
        int countNeurons = 0;
        ifs.read(reinterpret_cast<char*>(&countNeurons), sizeof(countNeurons));
        nn.pushLayer(countNeurons);
    }

    inputs.statusBar->setText(L"Reading neural network from \""s + filename + L"\"..."s);
    inputs.statusBar->redraw();

    for (size_t i = 0; i < lCount - 1; i++)
    {
        size_t countWeights = 0;
        ifs.read(reinterpret_cast<char*>(&countWeights), sizeof(countWeights));

        std::vector<double> vWeight;
        for (size_t j = 0; j < countWeights; j++)
        {
            double weight = 0;
            ifs.read(reinterpret_cast<char*>(&weight), sizeof(weight));
            vWeight.emplace_back(weight);
        }

        nn.setupWeights(i ,i + 1, vWeight);
        panels.window->processMessages();
    }

    ifs.close();

    inputs.statusBar->setText(L"Reading is completed!"s);
    inputs.statusBar->redraw();

    std::wstringstream ss;
    auto layers = nn.getLayers();
    for (size_t i = 0; i < layers.size(); i++)
    {
        ss << layers[i];
        if (i != layers.size() - 1)
            ss << L',';
    }

    inputs.layers->setText(ss.str());
    inputs.layers->redraw();
};

auto onSaveNNDataClick = [](const Awincs::Component::Point& p)
{
    std::wstring filename = inputs.saveNeuralNetwork->getText();
    std::ofstream ofs(filename, std::ios_base::binary);

    if (!ofs)
    {
        MessageBox(NULL, L"Failed to create neural network file", L"Neural network create failed!", MB_OK | MB_ICONWARNING);
        return;
    }

    const auto layers = nn.getLayers();
    size_t lCount = layers.size();

    ofs.write(reinterpret_cast<char*>(&lCount), sizeof(lCount));

    for (const auto& lSize : layers)
        ofs.write(reinterpret_cast<const char*>(&lSize), sizeof(lSize));

    const auto& vWeights = nn.getWeights();

    inputs.statusBar->setText(L"Writing neural network to \""s + filename + L"\"..."s);
    inputs.statusBar->redraw();

    for (const auto& vWeight : vWeights)
    {
        size_t vecSize = vWeight.size();
        ofs.write(reinterpret_cast<char*>(&vecSize), sizeof(vecSize));

        for (const auto& weight : vWeight)
            ofs.write(reinterpret_cast<const char*>(&weight), sizeof(weight));

        panels.window->processMessages();
    }
    ofs.close();

    inputs.statusBar->setText(L"Writting is completed!"s);
    inputs.statusBar->redraw();
};

auto onLoadTrainingDataClick = [](const Awincs::Component::Point& p)
{
    if (!inputs.loadTrainingData)
        return;

    if (!inputs.layers)
        return;

    std::wstring filename = inputs.loadTrainingData->getText();
    std::wifstream ifs(filename);

    if (!ifs)
        MessageBox(NULL, L"Failed to open training data file", L"Traing data open failed!", MB_OK | MB_ICONWARNING);

    const auto layers = parseVectorFromString<int>(inputs.layers->getText());
    const int countInputNeurons = layers.front();
    const int countOutputNeurons = layers.back();

    /* Reading training set */
    std::vector<std::pair<std::vector<double>, std::vector<double>>> trainingSet;
    std::wstring line;
    int lineNumber = 1;
    while (std::getline(ifs, line))
    {
        size_t index = line.find_first_of(L' ');
        auto cls = parseVectorFromString<double>(line.substr(0, index));
        index += 1;
        auto vec = parseVectorFromString<double>(line.substr(index));

        if (cls.size() != countOutputNeurons)
            DCONSOLE(L"Output neurons count missmatch! Actual: " << cls.size() << L"; expected: " 
                << countOutputNeurons << L". File: " << filename << L"; Line: " << lineNumber << L"\n");

        if (vec.size() != countInputNeurons)
            DCONSOLE(L"Input neurons count missmatch! Actual: " << vec.size() << L"; expected: "
                << countInputNeurons << L". File: " << filename << L"; Line: " << lineNumber << L"\n");

        trainingSet.push_back({ vec, cls });
        panels.window->processMessages();
        lineNumber++;
    }
    ifs.close();

    /* Setupping Neural Network */
    setupNeuralNetwork(layers, true);

    /* Trainning */
    const int countIterations = 10000;

    for (int i = 0; i < countIterations; i++)
    {
        for (const auto& ts : trainingSet)
        {
            panels.window->processMessages();
            nn.train(ts.first, ts.second);
        }

        if (i % 10)
        {
            inputs.statusBar->setText(L"Training iteration: "s + std::to_wstring(i));
            inputs.statusBar->redraw();
        }
    }

    inputs.statusBar->setText(L"Training completed!");
    inputs.statusBar->redraw();
};

/*********************************************************/
/*********************************************************/



/*********************************************************/
/*                   Classification                      */
auto onClassifyClick = [](const Awincs::Component::Point& p)
{
    if (!inputs.layers)
        return;

    if (!inputs.classifyOutput)
        return;

    if (!inputs.classify)
        return;

    setupNeuralNetwork(parseVectorFromString<int>(inputs.layers->getText()));

    auto vec = parseVectorFromString<double>(inputs.classify->getText());

    auto classification = nn.classify(vec);

    auto maxIterator = std::max_element(classification.begin(), classification.end());
    auto classIndex = std::distance(classification.begin(), maxIterator);
    std::wstringstream ss;
    ss << classIndex << L" class";

    if (inputs.classifyOutput)
    {
        inputs.classifyOutput->setText(ss.str());
        inputs.classifyOutput->redraw();
    }
};

/*********************************************************/
/*********************************************************/


/*********************************************************/
/*                      UISetup                          */

ButtonRef createButton(Awincs::Component::Point a, std::wstring content)
{
    const int width = 100;
    const int height = 30;

    auto button = std::make_shared<Awincs::ButtonComponent>();
    button->setAnchorPoint(a);
    button->setDimensions({width, height});
    button->setText(content);
    button->setBackgroundColor(Awincs::makeARGB(222, 222, 222), Awincs::ComponentState::DEFAULT);
    button->setBackgroundColor(Awincs::makeARGB(163, 163, 163), Awincs::ComponentState::HOVER);
    button->setBackgroundColor(Awincs::makeARGB(163, 163, 163), Awincs::ComponentState::ACTIVE);
    button->setTextColor(Awincs::makeARGB(163, 163, 163), Awincs::ComponentState::DEFAULT);
    button->setTextColor(Awincs::makeARGB(110, 110, 110), Awincs::ComponentState::HOVER);
    button->setTextColor(Awincs::makeARGB(110, 110, 110), Awincs::ComponentState::ACTIVE);

    return button;
}

void setupTitleBar(const WindowRef& wnd)
{
    auto titleBar = std::make_shared<Awincs::TitleBarComponent>();
    titleBar->setDimensions({ 360, 30 });
    titleBar->setText(MAIN_NN_PANEL_TITLE_BAR);
    titleBar->showText();
    titleBar->setParent(wnd);
    titleBar->setBackgroundColor(Awincs::makeARGB(222, 222, 222), Awincs::ComponentState::DEFAULT);
    titleBar->setBackgroundColor(Awincs::makeARGB(222, 222, 222), Awincs::ComponentState::HOVER);
    titleBar->setBackgroundColor(Awincs::makeARGB(222, 222, 222), Awincs::ComponentState::ACTIVE);
    titleBar->setTextColor(Awincs::makeARGB(163, 163, 163), Awincs::ComponentState::DEFAULT);
    titleBar->setTextColor(Awincs::makeARGB(110, 110, 110), Awincs::ComponentState::HOVER);
    titleBar->setTextColor(Awincs::makeARGB(110, 110, 110), Awincs::ComponentState::ACTIVE);
    auto cb = titleBar->getCloseButton();
    cb->setCrossColor(Awincs::makeARGB(222, 62, 27), Awincs::ComponentState::DEFAULT);
    cb->setCrossColor(Awincs::makeARGB(222, 222, 222), Awincs::ComponentState::HOVER);
    cb->setCrossColor(Awincs::makeARGB(222, 222, 222), Awincs::ComponentState::ACTIVE);
    cb->setBackgroundColor(Awincs::makeARGB(222, 62, 27), Awincs::ComponentState::HOVER);
    cb->setBackgroundColor(Awincs::makeARGB(222, 62, 27), Awincs::ComponentState::ACTIVE);
    panels.titleBar = titleBar;
}

InputRef setupInputDataRow(const ComponentRef& cm, int height, std::wstring name)
{
    auto label = std::make_shared<Awincs::PanelComponent>();
    label->setText(name);
    label->setDimensions({ 70, 30 });
    label->setAnchorPoint({ 10, height });
    label->showText();
    label->setParent(cm);
    label->setTextAlignment(Gdiplus::StringAlignment::StringAlignmentFar, Gdiplus::StringAlignment::StringAlignmentCenter);
    label->setTextAnchorPoint({ 70, 15 });

    auto input = std::make_shared<Awincs::InputComponent>();
    input->setDimensions({ 250, 30 });
    input->setAnchorPoint({ 100, height });
    input->setParent(cm);
    
    return input;
}


ComponentRef setupMainPanel(const ComponentRef& cmp, Awincs::Component::Point a, Awincs::Component::Dimensions d)
{
    auto panel = std::make_shared<Awincs::PanelComponent>();
    panel->setDimensions(d);
    panel->setAnchorPoint(a);
    panel->setParent(cmp);

    auto layersInput = setupInputDataRow(panel, 10, L"Layers:");
    inputs.layers = layersInput;

    auto loadButton = createButton({10, 50}, L"Load");
    loadButton->setParent(panel);
    loadButton->onClick(onLoadNNClick);
    auto saveButton = createButton({ 120, 50 }, L"Save");
    saveButton->setParent(panel);
    saveButton->onClick(onSaveNNClick);
    auto trainButton = createButton({ 230, 50 }, L"Train");
    trainButton->setParent(panel);
    trainButton->onClick(onTrainNNClick);

    auto classifyInput = setupInputDataRow(panel, 90, L"Classify: ");
    inputs.classify = classifyInput;

    auto classifyButton = createButton({ 10, 130 }, L"Classify");
    classifyButton->setParent(panel);
    classifyButton->onClick(onClassifyClick);

    auto answerLabel = std::make_shared<Awincs::PanelComponent>();
    answerLabel->setDimensions({ 100, 30 });
    answerLabel->setAnchorPoint({120, 130});
    answerLabel->setParent(panel);
    answerLabel->showText();
    answerLabel->setText(L"<undefined class>");
    answerLabel->setTextAlignment(Gdiplus::StringAlignment::StringAlignmentCenter, Gdiplus::StringAlignment::StringAlignmentCenter);
    answerLabel->setTextAnchorPoint({50, 15});
    inputs.classifyOutput = answerLabel;

    return panel;
}

ComponentRef setupStatusBar(const ComponentRef& cmp)
{
    auto [width, height] = cmp->getDimensions();
    const int panelHeight = 30;
    const int leftPadding = 10;

    auto panel = std::make_shared<Awincs::PanelComponent>();
    panel->setDimensions({ width, panelHeight });
    panel->setAnchorPoint({ 0, height - panelHeight });
    panel->setParent(cmp);
    panel->showText();
    panel->setText(L"Status: ok.");
    panel->setTextAlignment(Gdiplus::StringAlignment::StringAlignmentNear, Gdiplus::StringAlignment::StringAlignmentCenter);
    panel->setTextAnchorPoint({ leftPadding, panelHeight /2 });
    panel->setBackgroundColor(Awincs::makeARGB(222, 222, 222), Awincs::ComponentState::DEFAULT);
    panel->setBackgroundColor(Awincs::makeARGB(166, 166, 166), Awincs::ComponentState::HOVER);
    panel->setBackgroundColor(Awincs::makeARGB(166, 166, 166), Awincs::ComponentState::ACTIVE);
    panel->setTextColor(Awincs::makeARGB(163, 163, 163), Awincs::ComponentState::DEFAULT);
    panel->setTextColor(Awincs::makeARGB(110, 110, 110), Awincs::ComponentState::HOVER);
    panel->setTextColor(Awincs::makeARGB(110, 110, 110), Awincs::ComponentState::ACTIVE);
    inputs.statusBar = panel;

    return panel;
}

ComponentRef setupLoadPanel(const ComponentRef& cmp, Awincs::Component::Point a, Awincs::Component::Dimensions d)
{
    auto panel = std::make_shared<Awincs::PanelComponent>();
    panel->setDimensions(d);
    panel->setAnchorPoint(a);
    panel->setParent(cmp);

    auto loadFilpath = setupInputDataRow(panel, 10, L"Filepath: ");
    inputs.loadNeuralNetwork = loadFilpath;

    auto loadButton = createButton({10, 50}, L"Load");
    loadButton->setParent(panel);
    loadButton->onClick(onLoadNNDataClick);

    auto backButton = createButton({ 120, 50 }, L"Back");
    backButton->setParent(panel);
    backButton->onClick(onBackClick);

    return panel;
}

ComponentRef setupSavePanel(const ComponentRef& cmp, Awincs::Component::Point a, Awincs::Component::Dimensions d)
{
    auto panel = std::make_shared<Awincs::PanelComponent>();
    panel->setDimensions(d);
    panel->setAnchorPoint(a);
    panel->setParent(cmp);

    auto saveFilepath = setupInputDataRow(panel, 10, L"Filepath: ");
    inputs.saveNeuralNetwork = saveFilepath;

    auto saveButton = createButton({ 10, 50 }, L"Save");
    saveButton->setParent(panel);
    saveButton->onClick(onSaveNNDataClick);

    auto backButton = createButton({ 120, 50 }, L"Back");
    backButton->setParent(panel);
    backButton->onClick(onBackClick);

    return panel;
}

ComponentRef setupTrainPanel(const ComponentRef& cmp, Awincs::Component::Point a, Awincs::Component::Dimensions d)
{
    auto panel = std::make_shared<Awincs::PanelComponent>();
    panel->setDimensions(d);
    panel->setAnchorPoint(a);
    panel->setParent(cmp);

    auto filepathInput = setupInputDataRow(panel, 10, L"Filepath: ");
    inputs.loadTrainingData = filepathInput;

    auto loadButton = createButton({ 10, 50 }, L"Train");
    loadButton->setParent(panel);
    loadButton->onClick(onLoadTrainingDataClick);

    auto backButton = createButton({ 120, 50 }, L"Back");
    backButton->setParent(panel);
    backButton->onClick(onBackClick);

    return panel;
}

void mountPanels(const ComponentRef& cmp, Awincs::Component::Dimensions d)
{
    auto mainPanel = setupMainPanel(cmp, { 0, 30 }, d);
    panels.mainNNPanel = mainPanel;
    auto savePanel = setupSavePanel(cmp, { 0, 30 }, d);
    savePanel->unsetParent();
    panels.saveNNPanel = savePanel;
    auto loadPanel = setupLoadPanel(cmp, { 0, 30 }, d);
    loadPanel->unsetParent();
    panels.loadNNPanel = loadPanel;
    auto trainPanel = setupTrainPanel(cmp, { 0, 30 }, d);
    trainPanel->unsetParent();
    panels.trainNNPanel = trainPanel;
}

Awincs::AppRetType Awincs::App(std::vector<std::wstring> args)
{
    auto wnd = std::make_shared<WindowController>();
    wnd->setDimensions({ 360, 300 });
    wnd->setMinDimensions({360, 300});
    wnd->setMaxDimensions({360, 300});
    wnd->setAnchorPoint({ 200, 200 });
    wnd->setTitle(L"Neural Network");
    panels.window = wnd;

    setupTitleBar(wnd);
    mountPanels(wnd, {360, 240});
    setupStatusBar(wnd);

    wnd->redraw();
    wnd->show();
    return { wnd };
}

/*********************************************************/
/*********************************************************/