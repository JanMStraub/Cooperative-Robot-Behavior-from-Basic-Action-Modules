using System.Collections.Generic;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using UnityEngine;

/// <summary>
/// ML-Agents Robot Agent logic for controlling and learning robotic movement.
/// </summary>
public class RobotAgent : Agent
{
    private SimulationManager _simulationManagerInstance;
    private RobotManager _robotManagerInstance;
    private RobotController _robotController;

    [SerializeField] private Material winMaterial;
    [SerializeField] private Material loseMaterial;
    [SerializeField] private MeshRenderer signalMeshRenderer;

    [Header("Reward Tracking")]
    private readonly Dictionary<string, float> _rewardsByType = new();
    private readonly Dictionary<string, float> _currentObservations = new();
    private float[] _previousJointVelocities;
    private int _currentEpisodeSteps;
    private float _previousDistanceToGoal;

    private const int MaxStepsPerEpisode = 2000;

    public delegate void EpisodeEndHandler(RobotAgent agent, int finalStepCount);
    public event EpisodeEndHandler OnEpisodeEnd;

    private void Start()
    {
        _simulationManagerInstance = SimulationManager.Instance;
        _robotManagerInstance = RobotManager.Instance;
        _robotController = GetComponent<RobotController>();
    }

    public override void OnEpisodeBegin()
    {
        _currentEpisodeSteps = 0;
        _rewardsByType.Clear();
        _previousJointVelocities = new float[_robotController.robotJoints.Length];
        SetRandomStartingPositions();

        _robotController.SetDriveTargetsToZero();
        _previousDistanceToGoal = _robotController.GetDistanceToTarget();
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        _currentObservations.Clear();

        Vector3 baseToGripper = _robotController.robotGripperBase.localPosition - _robotController.robotJoints[0].transform.localPosition;
        Vector3 targetToGripper = _robotController.robotGripperBase.localPosition - _robotController.target.localPosition;

        AddObservation("BaseToGripper_X", baseToGripper.x, sensor);
        AddObservation("BaseToGripper_Y", baseToGripper.y, sensor);
        AddObservation("BaseToGripper_Z", baseToGripper.z, sensor);

        AddObservation("TargetToGripper_X", targetToGripper.x, sensor);
        AddObservation("TargetToGripper_Y", targetToGripper.y, sensor);
        AddObservation("TargetToGripper_Z", targetToGripper.z, sensor);

        foreach (var joint in _robotController.robotJoints)
        {
            AddObservation($"Joint{joint.index}_Angle", joint.jointPosition[0], sensor);
            AddObservation($"Joint{joint.index}_Velocity", joint.jointVelocity[0], sensor);
        }
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        var continuousActions = actions.ContinuousActions;

        for (int i = 0; i < _robotController.robotJoints.Length; i++)
        {
            float normalizedAction = Mathf.Clamp(continuousActions[i], -1f, 1f);
            _robotController.SetDriveTargets(i, normalizedAction);
        }

        float currentDistance = _robotController.GetDistanceToTarget();

        if (_currentEpisodeSteps > 0)
        {
            float progress = _previousDistanceToGoal - currentDistance;
            if (progress > 0) // Not sure if that is the right call
                AddRewardWithType("DistanceProgress", progress * 10f);
        }
        float logProximity = Mathf.Log(1f + (1f / (currentDistance + 1e-5f)));  // prevent divide-by-zero
        AddRewardWithType("Proximity", logProximity * 0.01f);
        AddRewardWithType("TimePenalty", -0.001f);
        AddRewardWithType("SmoothnessPenalty", -CalculateJointVelocityChangePenalty() * 0.001f);

        _previousDistanceToGoal = currentDistance;
        _currentEpisodeSteps++;

        if (currentDistance < _robotManagerInstance.convergenceThreshold)
        {
            AddRewardWithType("GoalReached", 10f);
            signalMeshRenderer.material = winMaterial;
            EndEpisode();
            return;
        }

        if (_currentEpisodeSteps >= MaxStepsPerEpisode)
        {
            AddRewardWithType("Timeout", -1f);
            signalMeshRenderer.material = loseMaterial;
            EndEpisode();
        }
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {

        Debug.Log("Heuristic update");
        var continuousActionsOut = actionsOut.ContinuousActions;

        for (int i = 0; i < _robotController.ArticulationBodyTargets.Length; i++)
        {
            continuousActionsOut[i] = _robotController.ArticulationBodyTargets[i];
            float normalizedAction = Mathf.Clamp(continuousActionsOut[i], -1f, 1f);
            _robotController.SetDriveTargets(i, normalizedAction);
            Debug.Log($"continuousActionsOut:  {continuousActionsOut[i]}");
        }
    }

    private void AddObservation(string name, float value, VectorSensor sensor)
    {
        _currentObservations[name] = value;
        sensor.AddObservation(value);
    }

    private float CalculateJointVelocityChangePenalty()
    {
        float penalty = 0f;

        for (int i = 0; i < _robotController.robotJoints.Length; i++)
        {
            var joint = _robotController.robotJoints[i];
            float currentVelocity = joint.jointVelocity[0];
            float delta = Mathf.Abs(currentVelocity - _previousJointVelocities[i]);
            penalty += delta;

            _previousJointVelocities[i] = currentVelocity; // Update for next frame
        }

        return penalty;
    }

    private void AddRewardWithType(string type, float reward)
    {
        _rewardsByType[type] = _rewardsByType.TryGetValue(type, out float current) ? current + reward : reward;
        AddReward(reward);
    }

    private void SetRandomStartingPositions()
    {
        Transform target = _robotController.target;

        Vector3 newPosition = target.localPosition;

        switch (target.name)
        {
            case "TargetCubeLeft":
                newPosition = new Vector3(Random.Range(0f, 0.3f), 0.1f, Random.Range(-0.4f, 0.4f));
                break;

            case "TargetCubeRight":
                newPosition = new Vector3(Random.Range(-0.3f, 0f), 0.1f, Random.Range(-0.4f, 0.4f));
                break;

            default:
                Debug.LogWarning($"Unknown target name: {target.name}. Position not randomized.");
                break;
        }

        target.localPosition = newPosition;

        _robotController.SetDriveTargetsToRandom();  // Reset joints
    }

    public new void EndEpisode()
    {
        OnEpisodeEnd?.Invoke(this, _currentEpisodeSteps);
        base.EndEpisode();
    }

    public int GetCurrentEpisodeSteps() => _currentEpisodeSteps;
    public int GetMaxStepsPerEpisode() => MaxStepsPerEpisode;
    public Dictionary<string, float> GetCurrentObservations() => new(_currentObservations);
    public Dictionary<string, float> GetRewardsByType() => new(_rewardsByType);
}

