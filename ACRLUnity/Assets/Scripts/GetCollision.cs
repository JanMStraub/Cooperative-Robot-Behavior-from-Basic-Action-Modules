using UnityEngine;

public class GetCollision : MonoBehaviour
{
    private RobotManager _robotManagerInstance;

    private void OnTriggerEnter(Collider other)
    {
        Collider triggerCollider = GetComponent<Collider>();

        if (triggerCollider != null)
        {
            Vector3 closestPoint = triggerCollider.ClosestPoint(other.transform.position);

            other.GetComponent<RobotController>().SetTargetReached(true);
            if (_robotManagerInstance.robotSpeed != 3.0f)
                _robotManagerInstance.SetRobotSpeed(3.0f);

            Debug.DrawRay(closestPoint, Vector3.up * 0.5f, Color.green, 100.0f);
        }
    }

    private void Start()
    {
        _robotManagerInstance = RobotManager.Instance;
    }
}
