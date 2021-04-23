#pragma once
#include <string>
#include <vector>

///
enum class ObjectTypes
{
	obj_person,
	obj_bicycle,
	obj_car,
	obj_motorbike,
	obj_aeroplane,
	obj_bus,
	obj_train,
	obj_truck,
	obj_boat,
	obj_traffic_light,
	obj_fire_hydrant,
	obj_stop_sign,
	obj_parking_meter,
	obj_bench,
	obj_bird,
	obj_cat,
	obj_dog,
	obj_horse,
	obj_sheep,
	obj_cow,
	obj_elephant,
	obj_bear,
	obj_zebra,
	obj_giraffe,
	obj_backpack,
	obj_umbrella,
	obj_handbag,
	obj_tie,
	obj_suitcase,
	obj_frisbee,
	obj_skis,
	obj_snowboard,
	obj_sports_ball,
	obj_kite,
	obj_baseball_bat,
	obj_baseball_glove,
	obj_skateboard,
	obj_surfboard,
	obj_tennis_racket,
	obj_bottle,
	obj_wine_glass,
	obj_cup,
	obj_fork,
	obj_knife,
	obj_spoon,
	obj_bowl,
	obj_banana,
	obj_apple,
	obj_sandwich,
	obj_orange,
	obj_broccoli,
	obj_carrot,
	obj_hot_dog,
	obj_pizza,
	obj_donut,
	obj_cake,
	obj_chair,
	obj_sofa,
	obj_pottedplant,
	obj_bed,
	obj_diningtable,
	obj_toilet,
	obj_tvmonitor,
	obj_laptop,
	obj_mouse,
	obj_remote,
	obj_keyboard,
	obj_cell_phone,
	obj_microwave,
	obj_oven,
	obj_toaster,
	obj_sink,
	obj_refrigerator,
	obj_book,
	obj_clock,
	obj_vase,
	obj_scissors,
	obj_teddy_bear,
	obj_hair_drier,
	obj_toothbrush,
    obj_vehicle,
	TypesCount
};

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
	static objtype_t Str2Type(const std::string& str)
	{
		for (size_t i = 0; i < m_typeNames.size(); ++i)
		{
			if (str == m_typeNames[i])
				return static_cast<objtype_t>(i);
		}
        m_typeNames.emplace_back(str);
		return static_cast<objtype_t>(m_typeNames.size()) - 1;
	}

private:
	static std::vector<std::string> m_typeNames;
	static std::string m_badTypeName;
};
