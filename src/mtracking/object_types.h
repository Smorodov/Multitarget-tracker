#pragma once
#include <mutex>
#include <string>
#include <vector>
#include <iostream>

typedef int objtype_t;
constexpr objtype_t bad_type = -1;

// The registry lives in its own (optionally shared) library, so its static
// data members must be explicitly imported/exported on Windows (roadmap M21).
#if defined(_WIN32) && defined(MT_CORE_SHARED)
#  if defined(mtracking_core_EXPORTS)
#    define MT_CORE_API __declspec(dllexport)
#  else
#    define MT_CORE_API __declspec(dllimport)
#  endif
#else
#  define MT_CORE_API
#endif

///
/// \brief Global registry of object type names.
///
/// The registry is seeded with the default schema (COCO-80 + "vehicle" +
/// "face") during static initialization.  After startup it is read-only:
/// Type2Str/Str2Type are pure lookups that never mutate the registry and
/// never index out of bounds (roadmap H22).  Custom class names that are
/// not part of the default schema are registered explicitly via
/// AddNewType during a (single-threaded) initialization phase; concurrent
/// reads after that are safe.
///
class TypeConverter
{
public:
	///
	/// \brief Bounds-checked type -> name lookup.
	/// \return the registered name, or m_badTypeName ("unknown") for
	///         bad_type and for any index outside the registry (roadmap H22).
	///
	static std::string Type2Str(objtype_t type)
	{
		if (type == bad_type || type < 0 ||
			static_cast<size_t>(type) >= m_typeNames.size())
		{
			return m_badTypeName;
		}
		return m_typeNames[static_cast<size_t>(type)];
	}

	///
	/// \brief Pure name -> type lookup; unknown names are NOT registered.
	/// \return the registered index, or bad_type when the name is unknown
	///         (roadmap H22).
	///
	static objtype_t Str2Type(const std::string& typeName)
	{
		for (size_t i = 0; i < m_typeNames.size(); ++i)
		{
			if (typeName == m_typeNames[i])
			{
				return static_cast<objtype_t>(i);
			}
		}
		return bad_type;
	}

	///
	/// \brief Explicitly register a type name.
	/// \return false when the name is empty or already registered, true when
	///         a new type was appended.  Thread-safe; call during
	///         initialization, not from the detection hot path.
	///
	static bool AddNewType(const std::string& typeName)
	{
		if (typeName.empty())
		{
			return false;
		}

		std::lock_guard<std::mutex> lock(m_mutex);
		for (size_t i = 0; i < m_typeNames.size(); ++i)
		{
			if (typeName == m_typeNames[i])
			{
				return false;
			}
		}
		m_typeNames.emplace_back(typeName);
		return true;
	}

	///
	/// \brief Number of registered types.
	///
	static size_t TypesCount()
	{
		return m_typeNames.size();
	}

private:
	static MT_CORE_API std::vector<std::string> m_typeNames;
	static MT_CORE_API std::string m_badTypeName;
	static MT_CORE_API std::mutex m_mutex;
};
