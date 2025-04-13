#pragma once
#include <string>
#include <vector>
#include <iostream>

typedef int objtype_t;
constexpr objtype_t bad_type = -1;

///
class TypeConverter
{
public:
	///
	static std::string Type2Str(objtype_t type)
	{
		return (type == bad_type) ? m_badTypeName : m_typeNames[type];
	}

	///
    static objtype_t Str2Type(const std::string& typeName)
	{
		for (size_t i = 0; i < m_typeNames.size(); ++i)
		{
            if (typeName == m_typeNames[i])
            {
                //std::cout << "Str2Type: " << typeName << " exist: " << i << std::endl;
				return static_cast<objtype_t>(i);
            }
		}
        m_typeNames.emplace_back(typeName);
        //std::cout << "Str2Type: " << typeName << " new: " << (m_typeNames.size()) - 1 << std::endl;
		return static_cast<objtype_t>(m_typeNames.size()) - 1;
	}

    static bool AddNewType(const std::string& typeName)
    {
        for (size_t i = 0; i < m_typeNames.size(); ++i)
        {
            if (typeName == m_typeNames[i])
            {
                //std::cout << "AddNewType: " << typeName << ": false" << std::endl;
                return false;
            }
        }
        m_typeNames.emplace_back(typeName);
        //std::cout << "AddNewType: " << typeName << ": " << (m_typeNames.size() - 1) << std::endl;
        return true;
    }

	static size_t TypesCount()
	{
		return m_typeNames.size();
	}

private:
	static std::vector<std::string> m_typeNames;
	static std::string m_badTypeName;
};
