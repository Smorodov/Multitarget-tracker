import sys
import glob
import getopt
import numpy as np
import cv2 as cv
import pymtracking as mt

print("OpenCV Version: {}".format(cv.__version__))


def draw_regions(img, regions, color):
    for reg in regions:
        cv.rectangle(img, (reg.x(), reg.y(), reg.width(), reg.height()), color, 2)


def draw_tracks(img, tracks, fps):
    for track in tracks:
        if track.isStatic:
            cv.rectangle(img, (track.x(), track.y(), track.width(), track.height()), (255, 0, 255), 2)
        elif track.IsRobust(int(fps / 4), 0.7, 0.1, 10.):
            cv.rectangle(img, (track.x(), track.y(), track.width(), track.height()), (0, 255, 0), 2)


def main():
    args, video_src = getopt.getopt(sys.argv[1:], '', ['cascade=', 'nested-cascade='])
    try:
        video_src = video_src[0]
    except:
        video_src = 0
    args = dict(args)

    cam = cv.VideoCapture(video_src)

    _ret, img = cam.read()

    fps = cam.get(cv.CAP_PROP_FPS)

    configBGFG = mt.MapStringString()
    configBGFG['samples'] = '20'
    configBGFG["pixelNeighbor"] = "3"
    configBGFG["distanceThreshold"] = "18"
    configBGFG["matchingThreshold"] = "3"
    configBGFG["updateFactor"] = "16"
    mdetector = mt.BaseDetector(mt.BaseDetector.Detectors.VIBE, configBGFG, img)
    mdetector.Init(configBGFG)
    mdetector.SetMinObjectSize(int(img.shape[0] / 100), int(img.shape[0] / 100))
    print(mdetector.CanGrayProcessing())

    tracker_settings = mt.TrackerSettings()

    tracker_settings.SetDistance(mt.MTracker.DistRects)
    tracker_settings.kalmanType = mt.MTracker.KalmanLinear
    tracker_settings.filterGoal = mt.MTracker.FilterCenter
    tracker_settings.lostTrackType = mt.MTracker.TrackCSRT
    tracker_settings.matchType = mt.MTracker.MatchHungrian
    tracker_settings.useAcceleration = False
    tracker_settings.dt = 0.2
    tracker_settings.accelNoiseMag = 0.2
    tracker_settings.distThres = 0.95
    tracker_settings.minAreaRadiusPix = -1.
    tracker_settings.minAreaRadiusK = 0.8
    tracker_settings.useAbandonedDetection = True
    tracker_settings.minStaticTime = 3
    tracker_settings.maxStaticTime = 3 * tracker_settings.minStaticTime
    tracker_settings.maximumAllowedSkippedFrames = int(tracker_settings.minStaticTime * fps)
    tracker_settings.maxTraceLength = 2 * tracker_settings.maximumAllowedSkippedFrames

    mtracker = mt.MTracker(tracker_settings)

    while True:
        _ret, img = cam.read()

        mdetector.Detect(img)
        regions = mdetector.GetDetects()

        mtracker.Update(regions, img, fps)
        tracks = mtracker.GetTracks()
        print("detects:", len(regions), ", tracks:", len(tracks))

        vis = img.copy()
        # draw_regions(vis, rects, (255, 0, 255))
        draw_tracks(vis, tracks, fps)
        cv.imshow('detect', vis)

        if cv.waitKey(1) == 27:
            break

    print('Done')


if __name__ == '__main__':
    main()
    cv.destroyAllWindows()
