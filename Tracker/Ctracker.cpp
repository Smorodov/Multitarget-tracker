#include "Ctracker.h"
using namespace cv;
using namespace std;

size_t CTrack::NextTrackID=0;
// ---------------------------------------------------------------------------
// Конструктор трека.
// При создании, трек начинается с какой то точки,
// эта точка и передается конструктору в качестве аргумента.
// ---------------------------------------------------------------------------
CTrack::CTrack(Point2f pt, float dt, float Accel_noise_mag)
{
	track_id=NextTrackID;

	NextTrackID++;
	// Каждый трек имеет свой фильтр Кальмана,
	// при помощи которого делается прогноз, где должна быть следующая точка.
	KF = new TKalmanFilter(pt,dt,Accel_noise_mag);
	// Здесь хранятся координаты точки, в которой трек прогнозирует следующее наблюдение (детект).
	prediction=pt;
	skipped_frames=0;
}
// ---------------------------------------------------------------------------
//
// ---------------------------------------------------------------------------
CTrack::~CTrack()
{
	// Освобождаем фильтр Кальмана.
	delete KF;
}

// ---------------------------------------------------------------------------
// Трекер. Производит управление треками. Создает, удаляет, уточняет.
// ---------------------------------------------------------------------------
CTracker::CTracker(float _dt, float _Accel_noise_mag, double _dist_thres, int _maximum_allowed_skipped_frames,int _max_trace_length)
{
dt=_dt;
Accel_noise_mag=_Accel_noise_mag;
dist_thres=_dist_thres;
maximum_allowed_skipped_frames=_maximum_allowed_skipped_frames;
max_trace_length=_max_trace_length;
}
// ---------------------------------------------------------------------------
//
// ---------------------------------------------------------------------------
void CTracker::Update(vector<Point2d>& detections)
{
	// -----------------------------------
	// Если треков еще нет, то начнем для каждой точки по треку
	// -----------------------------------
	if(tracks.size()==0)
	{
		// Если еще нет ни одного трека
		for(int i=0;i<detections.size();i++)
		{
			CTrack* tr=new CTrack(detections[i],dt,Accel_noise_mag);
			tracks.push_back(tr);
		}	
	}

	// -----------------------------------
	// Здесь треки уже есть в любом случае
	// -----------------------------------
	int N=tracks.size();		// треки
	int M=detections.size();	// детекты

	// Матрица расстояний от N-ного трека до M-ного детекта.
	vector< vector<double> > Cost(N,vector<double>(M));
	vector<int> assignment; // назначения

	// -----------------------------------
	// Треки уже есть, составим матрицу расстояний
	// -----------------------------------
	double dist;
	for(int i=0;i<tracks.size();i++)
	{	
		// Point2d prediction=tracks[i]->prediction;
		// cout << prediction << endl;
		for(int j=0;j<detections.size();j++)
		{
			Point2d diff=(tracks[i]->prediction-detections[j]);
			dist=sqrtf(diff.x*diff.x+diff.y*diff.y);
			Cost[i][j]=dist;
		}
	}
	// -----------------------------------
	// Решаем задачу о назначениях (треки и прогнозы фильтра)
	// -----------------------------------
	AssignmentProblemSolver APS;
	APS.Solve(Cost,assignment,AssignmentProblemSolver::optimal);

	// -----------------------------------
	// почистим assignment от пар с большим расстоянием
	// -----------------------------------
	// Не назначенные треки
	vector<int> not_assigned_tracks;

	for(int i=0;i<assignment.size();i++)
	{
		if(assignment[i]!=-1)
		{
			if(Cost[i][assignment[i]]>dist_thres)
			{
				assignment[i]=-1;
				// Отмечаем неназначенные треки, и увеличиваем счетчик пропущеных кадров,
				// когда количество пропущенных кадров превысит пороговое значение, трек стирается.
				not_assigned_tracks.push_back(i);
			}
		}
		else
		{			
			// Если треку не назначен детект, то увеличиваем счетчик пропущеных кадров.
			tracks[i]->skipped_frames++;
		}

	}

	// -----------------------------------
	// Если трек долго не получает детектов, удаляем
	// -----------------------------------
	for(int i=0;i<tracks.size();i++)
	{
		if(tracks[i]->skipped_frames>maximum_allowed_skipped_frames)
		{
			delete tracks[i];
			tracks.erase(tracks.begin()+i);
			assignment.erase(assignment.begin()+i);
			i--;
		}
	}
	// -----------------------------------
	// Выявляем неназначенные детекты
	// -----------------------------------
	vector<int> not_assigned_detections;
	vector<int>::iterator it;
	for(int i=0;i<detections.size();i++)
	{
		it=find(assignment.begin(), assignment.end(), i);
		if(it==assignment.end())
		{
			not_assigned_detections.push_back(i);
		}
	}

	// -----------------------------------
	// и начинаем для них новые треки
	// -----------------------------------
	if(not_assigned_detections.size()!=0)
	{
		for(int i=0;i<not_assigned_detections.size();i++)
		{
			CTrack* tr=new CTrack(detections[not_assigned_detections[i]],dt,Accel_noise_mag);
			tracks.push_back(tr);
		}	
	}

	// Апдейтим состояние фильтров

	for(int i=0;i<assignment.size();i++)
	{
		// Если трек апдейтился меньше одного раза, то состояние фильтра некорректно.

		tracks[i]->KF->GetPrediction();

		if(assignment[i]!=-1) // Если назначение есть то апдейтим по нему
		{
			tracks[i]->skipped_frames=0;
			tracks[i]->prediction=tracks[i]->KF->Update(detections[assignment[i]],1);
		}else				  // Если нет, то продолжаем прогнозировать
		{
			tracks[i]->prediction=tracks[i]->KF->Update(Point2f(0,0),0);	
		}
		
		if(tracks[i]->trace.size()>max_trace_length)
		{
			tracks[i]->trace.erase(tracks[i]->trace.begin(),tracks[i]->trace.end()-max_trace_length);
		}

		tracks[i]->trace.push_back(tracks[i]->prediction);
		tracks[i]->KF->LastResult=tracks[i]->prediction;
	}

}
// ---------------------------------------------------------------------------
//
// ---------------------------------------------------------------------------
CTracker::~CTracker(void)
{
	for(int i=0;i<tracks.size();i++)
	{
	delete tracks[i];
	}
	tracks.clear();
}
