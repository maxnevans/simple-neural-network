#include "pch.h"

#include <Awincs.h>
#include "NeuralNetwork.h"

using InputRef = std::shared_ptr<Awincs::InputComponent>;
using ButtonRef = std::shared_ptr<Awincs::ButtonComponent>;
using PanelRef = std::shared_ptr<Awincs::PanelComponent>;
using ComponentRef = std::shared_ptr<Awincs::Component>;
using WindowRef = std::shared_ptr<Awincs::WindowController>;
using TitleBarRef = std::shared_ptr<Awincs::TitleBarComponent>;

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
    ComponentRef window             = nullptr;
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
    // TODO
};

auto onSaveNNDataClick = [](const Awincs::Component::Point& p)
{
    // TODO
};

auto onLoadTrainingDataClick = [](const Awincs::Component::Point& p)
{
    // TODO
};

/*********************************************************/
/*********************************************************/



/*********************************************************/
/*                   Classification                      */
auto onClassifyClick = [](const Awincs::Component::Point& p)
{
    static std::vector<int> layers = {};

    if (auto newLayers = parseVectorFromString<int>(inputs.layers->getText()); newLayers != layers)
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

    return button;
}

void setupTitleBar(const WindowRef& wnd)
{
    auto titleBar = std::make_shared<Awincs::TitleBarComponent>();
    titleBar->setDimensions({ 360, 30 });
    titleBar->setText(MAIN_NN_PANEL_TITLE_BAR);
    titleBar->showText();
    titleBar->setParent(wnd);
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
    auto saveButton = createButton({ 110, 50 }, L"Save");
    saveButton->setParent(panel);
    saveButton->onClick(onSaveNNClick);
    auto trainButton = createButton({ 220, 50 }, L"Train");
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
    answerLabel->setText(L"1 class");
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
    inputs.statusBar = panel;

    return panel;
}

ComponentRef setupLoadPanel(const ComponentRef& cmp, Awincs::Component::Point a, Awincs::Component::Dimensions d)
{
    auto panel = std::make_shared<Awincs::PanelComponent>();
    panel->setDimensions(d);
    panel->setAnchorPoint(a);
    panel->setParent(cmp);

    setupInputDataRow(panel, 10, L"Filepath: ");

    auto loadButton = createButton({10, 50}, L"Load");
    loadButton->setParent(panel);

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

    setupInputDataRow(panel, 10, L"Filepath: ");

    auto saveButton = createButton({ 10, 50 }, L"Save");
    saveButton->setParent(panel);

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

    setupInputDataRow(panel, 10, L"Filepath: ");

    auto loadButton = createButton({ 10, 50 }, L"Load");
    loadButton->setParent(panel);

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