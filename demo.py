import sys
import glob
import getopt
import numpy as np
import cv2 as cv
import pymtracking as mt

print("OpenCV Version: {}".format(cv.__version__))


def draw_regions(img, regions, color):
    for reg in regions:
        brect = reg.brect
        cv.rectangle(img, (brect.x, brect.y, brect.width, brect.height), color, 2)


def draw_tracks(img, tracks, fps):
    for track in tracks:
        brect = track.GetBoundingRect()
        if track.isStatic:
            cv.rectangle(img, (brect.x, brect.y, brect.width, brect.height), (255, 0, 255), 2)
        elif track.IsRobust(int(fps / 4), 0.7, (0.1, 10.), 3):
            cv.rectangle(img, (brect.x, brect.y, brect.width, brect.height), (0, 255, 0), 2)
            trajectory = track.GetTrajectory()
            for i in range(0, len(trajectory) - 1):
                cv.line(img, trajectory[i], trajectory[i+1], (0, 255, 0), 1)


def main():
    args, video_src = getopt.getopt(sys.argv[1:], '', ['cascade=', 'nested-cascade='])
    try:
        video_src = video_src[0]
    except:
        video_src = 0
    args = dict(args)

    cam = cv.VideoCapture(video_src)

    _ret, img = cam.read()
    print("cam.read res = ", _ret, ", im size = ", img.shape)

    fps = cam.get(cv.CAP_PROP_FPS)
    print(video_src, " fps = ", fps)

    configBGFG = mt.KeyVal()
    configBGFG.Add('useRotatedRect', '20')
    configBGFG.Add('history', '1000')
    configBGFG.Add("nmixtures", "3")
    configBGFG.Add("backgroundRatio", "0.7")
    configBGFG.Add("noiseSigma", "0")
    print("configBGFG = ", configBGFG)
    mdetector = mt.BaseDetector(mt.BaseDetector.Detectors.MOG, configBGFG, img)
    print("CanGrayProcessing: ", mdetector.CanGrayProcessing())
    mdetector.SetMinObjectSize((1, 1))

    tracker_settings = mt.TrackerSettings()

    tracker_settings.SetDistance(mt.MTracker.DistRects)
    tracker_settings.kalmanType = mt.MTracker.KalmanLinear
    tracker_settings.filterGoal = mt.MTracker.FilterCenter
    tracker_settings.lostTrackType = mt.MTracker.TrackNone
    tracker_settings.matchType = mt.MTracker.MatchHungrian
    tracker_settings.useAcceleration = False
    tracker_settings.dt = 0.5
    tracker_settings.accelNoiseMag = 0.1
    tracker_settings.distThres = 0.95
    tracker_settings.minAreaRadiusPix = img.shape[0] / 5.
    tracker_settings.minAreaRadiusK = 0.8
    tracker_settings.useAbandonedDetection = False
    tracker_settings.maximumAllowedSkippedFrames = int(2 * fps)
    tracker_settings.maxTraceLength = int(2 * fps)

    mtracker = mt.MTracker(tracker_settings)

    while True:
        _ret, img = cam.read()
        if _ret:
            print("cam.read res = ", _ret, ", im size = ", img.shape, ", fps = ", fps)
        else:
            break

        mdetector.Detect(img)
        regions = mdetector.GetDetects()
        print("mdetector.Detect:", len(regions))

        mtracker.Update(regions, img, fps)
        tracks = mtracker.GetTracks()
        print("mtracker.Update:", len(tracks))

        vis = img.copy()
        # draw_regions(vis, regions, (255, 0, 255))
        draw_tracks(vis, tracks, fps)
        cv.imshow('detect', vis)

        if cv.waitKey(int(1000 / fps)) == 27:
            break

    print('Done')


if __name__ == '__main__':
    main()
    cv.destroyAllWindows()
