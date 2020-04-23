#pragma once

#include <string>

class Exception
{
public:
    Exception()
        :
        Exception(L"")
    {
    }
    Exception(std::wstring message)
    {
        this->message = message;
    }
    virtual std::wstring what() const noexcept
    {
        return message;
    }

private:
    std::wstring message;
};