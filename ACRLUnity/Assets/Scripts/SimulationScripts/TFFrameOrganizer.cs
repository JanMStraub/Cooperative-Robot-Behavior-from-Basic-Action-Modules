using UnityEngine;

namespace Simulation
{
    /// <summary>
    /// Organizes TF coordinate frame GameObjects created by the ROS-TCP-Connector
    /// TFSystem under a single "TF Frames" container in the scene hierarchy.
    ///
    /// The TFSystem creates one plain GameObject per ROS TF frame (e.g. base_link,
    /// gripper_base_link) at the scene root, which clutters the hierarchy at startup.
    /// This script subscribes to TFSystem's stream listener and reparents any root-level
    /// TF frame (no TF parent) into a dedicated container. Child frames are managed by
    /// TFSystem itself and will nest correctly once TF data establishes the tree.
    ///
    /// Self-bootstraps via RuntimeInitializeOnLoadMethod — no scene setup required.
    /// </summary>
    public class TFFrameOrganizer : MonoBehaviour
    {
        private GameObject _tfContainer;
        private const string _containerName = "TF Frames";

        /// <summary>
        /// Creates the container and subscribes to the TFSystem stream listener.
        /// notifyAllStreamsNow is true so frames created before this script starts
        /// (e.g. by earlier ROSConnection subscribers) are also organized.
        /// </summary>
        private void Start()
        {
            _tfContainer = new GameObject(_containerName);

            TFSystem tfSystem = TFSystem.GetOrCreateInstance();
            tfSystem.AddListener(OnTFStreamChanged, notifyAllStreamsNow: true);
        }

        /// <summary>
        /// Called by TFSystem whenever a stream is created or receives a TF update.
        /// Only moves frames that have no TF parent and are currently at the scene root.
        /// Frames with a TF parent are already nested inside another TF GameObject;
        /// leave those alone so TFSystem continues managing their Unity transform parent.
        /// </summary>
        private void OnTFStreamChanged(TFStream stream)
        {
            if (stream.Parent != null)
                return;

            GameObject go = stream.GameObject;
            if (go != null && go.transform.parent == null)
                go.transform.SetParent(_tfContainer.transform, worldPositionStays: true);
        }
    }
}
