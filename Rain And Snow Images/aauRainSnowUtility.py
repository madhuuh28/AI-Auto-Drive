import cv2
import numpy as np
import os

def registerPoints(points, 
                  homography, 
                  originDist, 
                  originCamMat, 
                  destinationDist, 
                  destinationCamMat):

    # Register points from one modality to another, e.g. RGB -> Thermal
    # by performing the following operations:
    # 1. Undistort the input points
    # 2. Transfer the undistorted input points to the desired modality
    #    by using the homography
    # 3. Distort the transfered points such that it fits on the output image

    undistortedRgbCoords = cv2.undistortPoints(np.array([points]).astype(float), 
                                                originCamMat, 
                                                originDist,
                                                np.eye(3, 3), originCamMat)

    destinationCoords = cv2.perspectiveTransform(undistortedRgbCoords, homography)

    # Distort points back to thermal
    # Begin by normalizing the points by using the thermal camera matrix
    normalizedDestinationCoords = []

    for destinationCoord in destinationCoords[0]:
        normCoord = [(destinationCoord[0] - destinationCamMat[0,2]) /  destinationCamMat[0,0],
                        (destinationCoord[1] - destinationCamMat[1,2]) /  destinationCamMat[1,1], 
                        1]
        normalizedDestinationCoords.append(normCoord)


    distDestinationCoords = cv2.projectPoints(np.array([normalizedDestinationCoords]).astype(np.float32), 
                                              np.array([0, 0, 0]).astype(np.float32), 
                                              np.array([0, 0, 0]).astype(np.float32), 
                                              destinationCamMat, 
                                              destinationDist)

    return distDestinationCoords[0] # The second dim are 3D coordinates we don't care about

def getRegistrationVarsFromFileName(imageFileName): 
    split = imageFileName.split('/')

    if len(split) >= 3:
        scene = split[0]
        sequence = split[1]

        # Then load the registration variables manually
        print("Reading calibration file at: " + os.path.join(scene, sequence + '-calib.yml'))
        fs = cv2.FileStorage(os.path.join(scene, sequence + '-calib.yml'), cv2.FILE_STORAGE_READ)

        registration = dict()
        registration["homCam1Cam2"] = fs.getNode("homCam1Cam2").mat()

        registration["homCam2Cam1"] = fs.getNode("homCam2Cam1").mat()
        registration["cam1CamMat"] = fs.getNode("cam1CamMat").mat()
        registration["cam2CamMat"] = fs.getNode("cam2CamMat").mat()

        registration["cam1DistCoeff"] = fs.getNode("cam1DistCoeff").mat()
        registration["cam2DistCoeff"] = fs.getNode("cam2DistCoeff").mat()

        return registration
    else:
        return None

def preparePoints(points):
    # A lot to do here if we are to look out for plenty of data types
    # For now, we will limit ourselves to look at points as provided
    # by COCO-compatible "segmentation" methods

    if type(points) is list:
        return np.reshape(points, (-1, 2))
    else:
        return NotImplementedError()

        

def registerRgbPointsToThermal(rgbPoints, rgbFileName):
    # Get the calibration file associated with the rgbFileName
    registration = getRegistrationVarsFromFileName(rgbFileName)
    rgbPoints = preparePoints(rgbPoints)

    thermalPoints = registerPoints(rgbPoints, 
                                   registration["homCam1Cam2"],
                                   registration["cam1DistCoeff"],
                                   registration["cam1CamMat"],
                                   registration["cam2DistCoeff"],
                                   registration["cam2CamMat"])

    return thermalPoints

def registerThermalPointsToRgb(thermalPoints, thermalFileName):
    # Get the calibration file associated with the rgbFileName
    registration = getRegistrationVarsFromFileName(thermalFileName)
    thermalPoints = preparePoints(thermalPoints)

    rgbPoints = registerPoints(thermalPoints, 
                                   registration["homCam2Cam1"],
                                   registration["cam2DistCoeff"],
                                   registration["cam2CamMat"],
                                   registration["cam1DistCoeff"],
                                   registration["cam1CamMat"])

    return rgbPoints