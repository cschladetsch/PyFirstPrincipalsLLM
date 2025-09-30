#include "imgui.h"
#include "imgui_impl_win32.h"
#include "imgui_impl_dx11.h"
#include <d3d11.h>
#include <tchar.h>
#include <vector>
#include <string>
#include <sstream>
#include <cmath>

// DirectX11 data
static ID3D11Device*            g_pd3dDevice = nullptr;
static ID3D11DeviceContext*     g_pd3dDeviceContext = nullptr;
static IDXGISwapChain*          g_pSwapChain = nullptr;
static ID3D11RenderTargetView*  g_mainRenderTargetView = nullptr;

// Forward declarations
bool CreateDeviceD3D(HWND hWnd);
void CleanupDeviceD3D();
void CreateRenderTarget();
void CleanupRenderTarget();
LRESULT WINAPI WndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

// RPN Calculator state
static std::vector<double> stack;
static char inputBuffer[256] = "";

void PushToStack(double value) {
    stack.push_back(value);
}

double PopFromStack() {
    if (stack.empty()) return 0.0;
    double value = stack.back();
    stack.pop_back();
    return value;
}

void PerformOperation(char op) {
    if (stack.size() < 2) return;

    double b = PopFromStack();
    double a = PopFromStack();

    switch (op) {
        case '+': PushToStack(a + b); break;
        case '-': PushToStack(a - b); break;
        case '*': PushToStack(a * b); break;
        case '/': if (b != 0) PushToStack(a / b); else PushToStack(0); break;
    }
}

// Main code
int main(int, char**)
{
    // Create application window
    WNDCLASSEXW wc = { sizeof(wc), CS_CLASSDC, WndProc, 0L, 0L, GetModuleHandle(nullptr), nullptr, nullptr, nullptr, nullptr, L"RPN Calculator", nullptr };
    ::RegisterClassExW(&wc);
    HWND hwnd = ::CreateWindowW(wc.lpszClassName, L"RPN Calculator", WS_OVERLAPPEDWINDOW, 100, 100, 500, 600, nullptr, nullptr, wc.hInstance, nullptr);

    // Initialize Direct3D
    if (!CreateDeviceD3D(hwnd))
    {
        CleanupDeviceD3D();
        ::UnregisterClassW(wc.lpszClassName, wc.hInstance);
        return 1;
    }

    // Show the window
    ::ShowWindow(hwnd, SW_SHOWDEFAULT);
    ::UpdateWindow(hwnd);

    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    // Setup Dear ImGui style
    ImGui::StyleColorsDark();

    // Setup Platform/Renderer backends
    ImGui_ImplWin32_Init(hwnd);
    ImGui_ImplDX11_Init(g_pd3dDevice, g_pd3dDeviceContext);

    // Main loop
    bool done = false;
    ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);

    while (!done)
    {
        // Poll and handle messages
        MSG msg;
        while (::PeekMessage(&msg, nullptr, 0U, 0U, PM_REMOVE))
        {
            ::TranslateMessage(&msg);
            ::DispatchMessage(&msg);
            if (msg.message == WM_QUIT)
                done = true;
        }
        if (done)
            break;

        // Start the Dear ImGui frame
        ImGui_ImplDX11_NewFrame();
        ImGui_ImplWin32_NewFrame();
        ImGui::NewFrame();

        // RPN Calculator Window
        {
            ImGui::SetNextWindowPos(ImVec2(0, 0));
            ImGui::SetNextWindowSize(io.DisplaySize);
            ImGui::Begin("RPN Calculator", nullptr, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse);

            ImGui::Text("Reverse Polish Notation Calculator");
            ImGui::Separator();

            // Display stack
            ImGui::Text("Stack:");
            ImGui::BeginChild("Stack", ImVec2(0, 200), true);
            for (int i = stack.size() - 1; i >= 0; i--) {
                ImGui::Text("%d: %.6f", i, stack[i]);
            }
            ImGui::EndChild();

            // Input field
            ImGui::Separator();
            ImGui::Text("Enter number or operator:");
            bool enterPressed = ImGui::InputText("##input", inputBuffer, IM_ARRAYSIZE(inputBuffer), ImGuiInputTextFlags_EnterReturnsTrue);

            if (enterPressed && strlen(inputBuffer) > 0) {
                std::string input(inputBuffer);

                // Check if it's an operator
                if (input == "+" || input == "-" || input == "*" || input == "/") {
                    PerformOperation(input[0]);
                }
                // Try to parse as number
                else {
                    try {
                        double value = std::stod(input);
                        PushToStack(value);
                    } catch (...) {
                        // Invalid input, ignore
                    }
                }

                // Clear input
                inputBuffer[0] = '\0';
                ImGui::SetKeyboardFocusHere(-1);
            }

            ImGui::Separator();

            // Button pad
            ImGui::Text("Number Pad:");
            for (int i = 7; i <= 9; i++) {
                if (i > 7) ImGui::SameLine();
                if (ImGui::Button(std::to_string(i).c_str(), ImVec2(50, 50))) {
                    std::string str = std::to_string(i);
                    strcat_s(inputBuffer, str.c_str());
                }
            }

            for (int i = 4; i <= 6; i++) {
                if (i > 4) ImGui::SameLine();
                if (ImGui::Button(std::to_string(i).c_str(), ImVec2(50, 50))) {
                    std::string str = std::to_string(i);
                    strcat_s(inputBuffer, str.c_str());
                }
            }

            for (int i = 1; i <= 3; i++) {
                if (i > 1) ImGui::SameLine();
                if (ImGui::Button(std::to_string(i).c_str(), ImVec2(50, 50))) {
                    std::string str = std::to_string(i);
                    strcat_s(inputBuffer, str.c_str());
                }
            }

            if (ImGui::Button("0", ImVec2(50, 50))) {
                strcat_s(inputBuffer, "0");
            }
            ImGui::SameLine();
            if (ImGui::Button(".", ImVec2(50, 50))) {
                strcat_s(inputBuffer, ".");
            }
            ImGui::SameLine();
            if (ImGui::Button("Enter", ImVec2(50, 50))) {
                if (strlen(inputBuffer) > 0) {
                    try {
                        double value = std::stod(inputBuffer);
                        PushToStack(value);
                        inputBuffer[0] = '\0';
                    } catch (...) {}
                }
            }

            ImGui::Separator();
            ImGui::Text("Operators:");

            if (ImGui::Button("+", ImVec2(50, 50))) {
                PerformOperation('+');
            }
            ImGui::SameLine();
            if (ImGui::Button("-", ImVec2(50, 50))) {
                PerformOperation('-');
            }
            ImGui::SameLine();
            if (ImGui::Button("*", ImVec2(50, 50))) {
                PerformOperation('*');
            }
            ImGui::SameLine();
            if (ImGui::Button("/", ImVec2(50, 50))) {
                PerformOperation('/');
            }

            ImGui::Separator();

            if (ImGui::Button("Clear Stack", ImVec2(120, 30))) {
                stack.clear();
            }
            ImGui::SameLine();
            if (ImGui::Button("Pop", ImVec2(120, 30))) {
                PopFromStack();
            }

            ImGui::End();
        }

        // Rendering
        ImGui::Render();
        const float clear_color_with_alpha[4] = { clear_color.x * clear_color.w, clear_color.y * clear_color.w, clear_color.z * clear_color.w, clear_color.w };
        g_pd3dDeviceContext->OMSetRenderTargets(1, &g_mainRenderTargetView, nullptr);
        g_pd3dDeviceContext->ClearRenderTargetView(g_mainRenderTargetView, clear_color_with_alpha);
        ImGui_ImplDX11_RenderDrawData(ImGui::GetDrawData());

        g_pSwapChain->Present(1, 0);
    }

    // Cleanup
    ImGui_ImplDX11_Shutdown();
    ImGui_ImplWin32_Shutdown();
    ImGui::DestroyContext();

    CleanupDeviceD3D();
    ::DestroyWindow(hwnd);
    ::UnregisterClassW(wc.lpszClassName, wc.hInstance);

    return 0;
}

// Helper functions to setup Direct3D

bool CreateDeviceD3D(HWND hWnd)
{
    DXGI_SWAP_CHAIN_DESC sd;
    ZeroMemory(&sd, sizeof(sd));
    sd.BufferCount = 2;
    sd.BufferDesc.Width = 0;
    sd.BufferDesc.Height = 0;
    sd.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    sd.BufferDesc.RefreshRate.Numerator = 60;
    sd.BufferDesc.RefreshRate.Denominator = 1;
    sd.Flags = DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH;
    sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    sd.OutputWindow = hWnd;
    sd.SampleDesc.Count = 1;
    sd.SampleDesc.Quality = 0;
    sd.Windowed = TRUE;
    sd.SwapEffect = DXGI_SWAP_EFFECT_DISCARD;

    UINT createDeviceFlags = 0;
    D3D_FEATURE_LEVEL featureLevel;
    const D3D_FEATURE_LEVEL featureLevelArray[2] = { D3D_FEATURE_LEVEL_11_0, D3D_FEATURE_LEVEL_10_0, };
    HRESULT res = D3D11CreateDeviceAndSwapChain(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, createDeviceFlags, featureLevelArray, 2, D3D11_SDK_VERSION, &sd, &g_pSwapChain, &g_pd3dDevice, &featureLevel, &g_pd3dDeviceContext);
    if (res == DXGI_ERROR_UNSUPPORTED)
        res = D3D11CreateDeviceAndSwapChain(nullptr, D3D_DRIVER_TYPE_WARP, nullptr, createDeviceFlags, featureLevelArray, 2, D3D11_SDK_VERSION, &sd, &g_pSwapChain, &g_pd3dDevice, &featureLevel, &g_pd3dDeviceContext);
    if (res != S_OK)
        return false;

    CreateRenderTarget();
    return true;
}

void CleanupDeviceD3D()
{
    CleanupRenderTarget();
    if (g_pSwapChain) { g_pSwapChain->Release(); g_pSwapChain = nullptr; }
    if (g_pd3dDeviceContext) { g_pd3dDeviceContext->Release(); g_pd3dDeviceContext = nullptr; }
    if (g_pd3dDevice) { g_pd3dDevice->Release(); g_pd3dDevice = nullptr; }
}

void CreateRenderTarget()
{
    ID3D11Texture2D* pBackBuffer;
    g_pSwapChain->GetBuffer(0, IID_PPV_ARGS(&pBackBuffer));
    g_pd3dDevice->CreateRenderTargetView(pBackBuffer, nullptr, &g_mainRenderTargetView);
    pBackBuffer->Release();
}

void CleanupRenderTarget()
{
    if (g_mainRenderTargetView) { g_mainRenderTargetView->Release(); g_mainRenderTargetView = nullptr; }
}

// Forward declare message handler from imgui_impl_win32.cpp
extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

// Win32 message handler
LRESULT WINAPI WndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    if (ImGui_ImplWin32_WndProcHandler(hWnd, msg, wParam, lParam))
        return true;

    switch (msg)
    {
    case WM_SIZE:
        if (g_pd3dDevice != nullptr && wParam != SIZE_MINIMIZED)
        {
            CleanupRenderTarget();
            g_pSwapChain->ResizeBuffers(0, (UINT)LOWORD(lParam), (UINT)HIWORD(lParam), DXGI_FORMAT_UNKNOWN, 0);
            CreateRenderTarget();
        }
        return 0;
    case WM_SYSCOMMAND:
        if ((wParam & 0xfff0) == SC_KEYMENU)
            return 0;
        break;
    case WM_DESTROY:
        ::PostQuitMessage(0);
        return 0;
    }
    return ::DefWindowProcW(hWnd, msg, wParam, lParam);
}