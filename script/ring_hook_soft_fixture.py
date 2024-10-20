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
        self.objectPos = [1.3, -.1, 3.4]
        self.objectQtn = [0, 1, 0, 1]
        self.objectEul = list(p.getEulerFromQuaternion(self.objectQtn))
        self.obstacle = 'Hook'
        self.obstaclePos = [0, 0, 2]
        self.obstacleEul = [1.57, -0.3, 0]
        self.obstacleQtn = list(p.getQuaternionFromEuler(self.obstacleEul))
        self.obstacleScale = [.1, .1, .1]
        self.startFrame = 280
        self.endFrame = 520
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


class RigidObjectCaging():
    '''
    Class to run energy-biased search for the caging analysis.
    '''
    def __init__(self, args, obj_start, obs_pos, obs_qtn):
        self.args = args
        self.obj_start = obj_start
        self.obs_pos = obs_pos
        self.obs_qtn = obs_qtn
        self.escapeEnergyCost = []

        # Pybullet setup
        p.connect(p.GUI)
        p.setTimeStep(1./240.)
        p.setRealTimeSimulation(0)

        # Create object and obstacle
        self.load_object()
        self.add_obstacles()

        # Initialize search parameters
        self.create_ompl_interface()

    def load_object(self):
        '''
        Load ring object into pybullet.
        '''
        self.paths = path_collector()
        self.object_id = p.loadURDF(self.paths[self.args.object], self.obj_start[:3], self.obj_start[3:])

    def add_obstacles(self):
        '''
        Add hook obstacle to pybullet.
        '''
        mesh_collision_shape = p.createCollisionShape(
            shapeType=p.GEOM_MESH,
            fileName=self.paths[self.args.obstacle],
            meshScale=[.1, .1, .1],
            flags=p.GEOM_FORCE_CONCAVE_TRIMESH,
        )
        self.obstacle_id = p.createMultiBody(
            baseCollisionShapeIndex=mesh_collision_shape,
            basePosition=self.obs_pos,
            baseOrientation=self.obs_qtn,
        )

    def create_ompl_interface(self):
        '''
        Create OMPL planner interface (simplified).
        '''
        # self.pb_ompl_interface = PbOMPL(self.robot, self.args, self.obstacles)
        while True:
            p.stepSimulation()
            p.setGravity(0, 0, -10)
            time.sleep(1/240.)
        # pass

    def energy_biased_search(self):
        '''
        Perform energy-biased search for caging analysis.
        '''
        # Simplified version, placeholder for actual search logic
        print("Running energy-biased search...")

        # Example result
        escape_energy_cost = 10.0  # Simplified, replace with actual computation
        self.escapeEnergyCost.append(escape_energy_cost)
        print(f"Escape energy cost: {escape_energy_cost}")


if __name__ == '__main__':
    class Args:
        object = 'Ring'
        obstacle = 'Hook'

    args = Args()

    # Run dynamic falling simulation
    sce = runScenario(args)
    sce.runDynamicFalling()

    # Perform caging analysis using the final frame's object and obstacle positions
    obj_start = sce.objBasePosSce[-1] + list(sce.objBaseQtnSce[-1])
    obs_pos = sce.obsBasePosSce[-1]
    obs_qtn = sce.obsBaseQtnSce[-1]

    caging = RigidObjectCaging(args, obj_start, obs_pos, obs_qtn)
    caging.energy_biased_search()
