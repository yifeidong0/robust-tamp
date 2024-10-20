import pybullet as p
import time
import os.path as osp
import sys
import pybullet_data
import numpy as np
import math

sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../'))
from script.utils import ObjectFromUrdf


def path_collector():
    '''
    Paths of all models.
    '''
    return {
            'Ring': 'models/ring/ring.urdf', 
            'Hook': 'models/triple_hook/triple_hook_vhacd.obj', 
            }


class runScenario():
    def __init__(self, args):
        # Pybullet setup
        p.connect(p.GUI)
        p.setTimeStep(1./240.)
        p.setRealTimeSimulation(0)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        self.paths = path_collector()
        self.args = args
        self.gravity = -9.81
        self.downsampleRate = 1

        # Load object and obstacle models
        self.initializeParams()
        self.loadObject()
        self.loadObstacle()

        # Data structures for object and obstacle configs
        self.objBasePosSce = []
        self.objBaseQtnSce = []
        self.obsBasePosSce = []
        self.obsBaseQtnSce = []
        self.idxSce = []

    def initializeParams(self):
        '''
        Initialize parameters for HookTrapsRing scenario.
        '''
        self.object = 'Ring'
        self.objectPos = [1.3,-.1,3.4]
        self.objectQtn = [0,1,0,1]
        self.objectEul = list(p.getEulerFromQuaternion(self.objectQtn))
        self.obstacle = 'Hook'
        self.obstaclePos = [0, 0, 2]
        self.obstacleEul = [1.57, -0.3, 0]
        self.obstacleQtn = list(p.getQuaternionFromEuler(self.obstacleEul))
        self.obstacleScale = [.1, .1, .1]
        self.basePosBounds = [[-.5, 2], [-.5, .5], [1.3, 2.7]]  # searching bounds
        self.goalSpaceBounds = [[1.4, 2], [-.5, .5], [1.3, 2.1]] + [[math.radians(-180), math.radians(180)]] + [[-.2, .2]]*2
        self.goalCoMPose = [1.6, 0, 1.5] + [1.57, 0, 0]
        self.startFrame = 280
        self.endFrame = 1520
        self.downsampleRate = 1

    def loadObject(self):
        '''
        Loading models of objects into pybullet.
        '''
        self.objectId = p.loadURDF(self.paths[self.args.object], self.objectPos, self.objectQtn)

    def loadObstacle(self):
        '''
        Loading models of obstacles into pybullet.
        '''
        mesh_collision_shape = p.createCollisionShape(
            shapeType=p.GEOM_MESH,
            fileName=self.paths[self.args.obstacle],
            meshScale=self.obstacleScale,
            flags=p.GEOM_FORCE_CONCAVE_TRIMESH,
        )
        self.obstacleId = p.createMultiBody(
            baseCollisionShapeIndex=mesh_collision_shape,
            basePosition=self.obstaclePos,
            baseOrientation=self.obstacleQtn,
        )
        self.obstacle = ObjectFromUrdf(self.obstacleId)

    def runDynamicFalling(self):
        '''
        Run the HookTrapsRing scenario for catching a falling ring with a hook.
        '''
        i = 0        
        while True:
            p.stepSimulation()
            p.setGravity(0, 0, self.gravity)
            time.sleep(1/240.)

            if i % self.downsampleRate == 0 and i >= self.startFrame:
                gemPos, gemQtn = p.getBasePositionAndOrientation(self.objectId)

                # Record object (ring) state
                self.objBasePosSce.append(list(gemPos))
                self.objBaseQtnSce.append(list(gemQtn))

                # Record obstacle (hook) state
                self.obsBasePosSce.append(self.obstaclePos)
                self.obsBaseQtnSce.append(self.obstacleQtn)
                self.idxSce.append(i)

            if i == self.endFrame:
                p.disconnect()
                break
            i += 1


if __name__ == '__main__':
    class Args:
        object = 'Ring'
        obstacle = 'Hook'
        scenario = 'HookTrapsRing'

    args = Args()
    sce = runScenario(args)
    sce.runDynamicFalling()
