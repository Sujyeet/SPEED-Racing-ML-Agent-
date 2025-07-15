using KartGame.KartSystems;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
using UnityEngine;
using Random = UnityEngine.Random;

namespace KartGame.AI
{
    [System.Serializable]
    public struct Sensor
    {
        public Transform Transform;
        public float RayDistance;
        public float HitValidationDistance;
    }

    public enum AgentMode
    {
        Training,
        Inferencing
    }

    public class KartAgent : Agent, IInput
    {
        #region Training Modes
        [Tooltip("Are we training the agent or is the agent production ready?")]
        public AgentMode Mode = AgentMode.Training;
        [Tooltip("What is the initial checkpoint the agent will go to? This value is only for inferencing.")]
        public ushort InitCheckpointIndex;
        #endregion

        #region Senses
        [Header("Observation Params")]
        [Tooltip("What objects should the raycasts hit and detect?")]
        public LayerMask Mask;
        [Tooltip("Sensors contain ray information to sense out the world, you can have as many sensors as you need.")]
        public Sensor[] Sensors;
        [Header("Checkpoints"), Tooltip("What are the series of checkpoints for the agent to seek and pass through?")]
        public Collider[] Colliders;
        [Tooltip("What layer are the checkpoints on? This should be an exclusive layer for the agent to use.")]
        public LayerMask CheckpointMask;
        [Space]
        [Tooltip("Would the agent need a custom transform to be able to raycast and hit the track? " +
                 "If not assigned, then the root transform will be used.")]
        public Transform AgentSensorTransform;
        #endregion

        #region Basic Rewards
        [Header("Basic Rewards"), Tooltip("What penalty is given when the agent crashes?")]
        public float HitPenalty = -1f;
        [Tooltip("How much reward is given when the agent successfully passes the checkpoints?")]
        public float PassCheckpointReward = 1f;
        [Tooltip("Should typically be a small value, but we reward the agent for moving in the right direction.")]
        public float TowardsCheckpointReward = 0.01f;
        [Tooltip("Typically if the agent moves faster, we want to reward it for finishing the track quickly.")]
        public float SpeedReward = 0.02f;
        [Tooltip("Reward the agent when it keeps accelerating")]
        public float AccelerationReward = 0.01f;
        #endregion

        #region Speed Thresholds
        [Header("Speed Thresholds")]
        [Tooltip("Minimum speed for speed rewards")]
        public float MinSpeedThreshold = 1.5f; // Reduced for 1x time scale
        [Tooltip("Maximum speed for speed rewards")]
        public float MaxSpeedThreshold = 6f; // Reduced for 1x time scale
        [Tooltip("Penalty for driving too slowly")]
        public float SlowSpeedPenalty = -0.01f;
        [Tooltip("Penalty for driving too fast")]
        public float FastSpeedPenalty = -0.005f;
        #endregion

        #region Natural Driving Rewards
        [Header("Natural Driving Rewards")]
        [Tooltip("Penalty for jerky steering movements")]
        public float SmoothSteeringPenalty = -0.001f;
        [Tooltip("Reward for maintaining appropriate speed on straights")]
        public float StraightSpeedReward = 0.005f;
        
        public float OptimalStraightSpeed = 8f; // Reduced for 1x time scale
        [Tooltip("Minimum speed for straight segments")]
        public float MinStraightSpeed = 4f; // Reduced for 1x time scale
        [Tooltip("Angle threshold to consider a segment as 'straight'")]
        public float StraightAngleThreshold = 15f;
        #endregion

        #region Time Scale Adjustment
        [Header("Time Scale Adjustment")]
        [Tooltip("Multiplier to adjust actions for different time scales (0.05 for 20x→1x)")]
        public float timeScaleMultiplier = 0.05f;
        [Tooltip("Enable/disable time scale adjustment")]
        public bool enableTimeScaleAdjustment = true;
        [Tooltip("Separate multiplier for steering sensitivity")]
        public float steeringMultiplier = 0.05f;
        [Tooltip("Separate multiplier for throttle/brake sensitivity")]
        public float throttleMultiplier = 0.05f;
        #endregion

        #region ResetParams
        [Header("Inference Reset Params")]
        [Tooltip("What is the unique mask that the agent should detect when it falls out of the track?")]
        public LayerMask OutOfBoundsMask;
        [Tooltip("What are the layers we want to detect for the track and the ground?")]
        public LayerMask TrackMask;
        [Tooltip("How far should the ray be when casted? For larger karts - this value should be larger too.")]
        public float GroundCastDistance;
        #endregion

        #region Debugging
        [Header("Debug Option")] [Tooltip("Should we visualize the rays that the agent draws?")]
        public bool ShowRaycasts;
        #endregion

        // Core variables
        ArcadeKart m_Kart;
        bool m_Acceleration;
        bool m_Brake;
        float m_Steering;
        int m_CheckpointIndex;
        bool m_EndEpisode;
        float m_LastAccumulatedReward;

        // Natural driving variables
        private float m_LastSteering = 0f;

        void Awake()
        {
            m_Kart = GetComponent<ArcadeKart>();
            if (AgentSensorTransform == null) AgentSensorTransform = transform;
        }

        void Start()
        {
            Rigidbody rb = GetComponent<Rigidbody>();
            
            // More natural physics settings
            rb.drag = 0.3f;
            rb.angularDrag = 3f;
            rb.mass = 1200f;
            rb.centerOfMass = new Vector3(0, -0.4f, 0.1f);
        }

        void Update()
        {
            if (m_EndEpisode)
            {
                m_EndEpisode = false;
                AddReward(m_LastAccumulatedReward);
                EndEpisode();
                OnEpisodeBegin();
            }
        }

        void LateUpdate()
        {
            switch (Mode)
            {
                case AgentMode.Inferencing:
                    if (ShowRaycasts) 
                        Debug.DrawRay(transform.position, Vector3.down * GroundCastDistance, Color.cyan);

                    if (Physics.Raycast(transform.position + Vector3.up, Vector3.down, out var hit, GroundCastDistance, TrackMask)
                        && ((1 << hit.collider.gameObject.layer) & OutOfBoundsMask) > 0)
                    {
                        var checkpoint = Colliders[m_CheckpointIndex].transform;
                        transform.localRotation = checkpoint.rotation;
                        transform.position = checkpoint.position;
                        m_Kart.Rigidbody.velocity = default;
                        m_Steering = 0f;
                        m_Acceleration = m_Brake = false; 
                    }
                    break;
            }
        }

        void OnTriggerEnter(Collider other)
        {
            var maskedValue = 1 << other.gameObject.layer;
            var triggered = maskedValue & CheckpointMask;

            FindCheckpointIndex(other, out var index);

            if (triggered > 0 && (index > m_CheckpointIndex || (index == 0 && m_CheckpointIndex == Colliders.Length - 1)))
            {
                AddReward(PassCheckpointReward);
                m_CheckpointIndex = index;
            }
        }

        void FindCheckpointIndex(Collider checkPoint, out int index)
        {
            for (int i = 0; i < Colliders.Length; i++)
            {
                if (Colliders[i].GetInstanceID() == checkPoint.GetInstanceID())
                {
                    index = i;
                    return;
                }
            }
            index = -1;
        }

        // Check if current segment is straight based on checkpoint angles
        bool IsOnStraightSegment()
        {
            if (Colliders.Length < 3) return true;

            var current = m_CheckpointIndex;
            var next = (current + 1) % Colliders.Length;
            var nextNext = (current + 2) % Colliders.Length;

            var dir1 = (Colliders[next].transform.position - Colliders[current].transform.position).normalized;
            var dir2 = (Colliders[nextNext].transform.position - Colliders[next].transform.position).normalized;

            float angle = Vector3.Angle(dir1, dir2);
            return angle < StraightAngleThreshold;
        }

        // Enhanced speed reward calculation with thresholds
        void CalculateSpeedRewards()
        {
            float currentSpeed = m_Kart.LocalSpeed();
            
            // Basic speed reward with thresholds
            if (currentSpeed >= MinSpeedThreshold && currentSpeed <= MaxSpeedThreshold)
            {
                // Reward proportional to speed within optimal range
                float speedRatio = (currentSpeed - MinSpeedThreshold) / (MaxSpeedThreshold - MinSpeedThreshold);
                AddReward(speedRatio * SpeedReward);
            }
            else if (currentSpeed < MinSpeedThreshold)
            {
                AddReward(SlowSpeedPenalty);
            }
            else if (currentSpeed > MaxSpeedThreshold)
            {
                AddReward(FastSpeedPenalty);
            }
        }

        // Natural driving reward functions
        void AddSmoothSteeringReward()
        {
            float steeringChange = Mathf.Abs(m_Steering - m_LastSteering);
            if (steeringChange > 0.3f) // Threshold for jerky movement
            {
                AddReward(SmoothSteeringPenalty * steeringChange);
            }
            m_LastSteering = m_Steering;
        }

        void AddStraightSegmentRewards()
        {
            if (IsOnStraightSegment())
            {
                float currentSpeed = m_Kart.LocalSpeed();
                
                // Reward for appropriate speed on straights
                if (currentSpeed >= MinStraightSpeed && currentSpeed <= OptimalStraightSpeed)
                {
                    float speedRatio = currentSpeed / OptimalStraightSpeed;
                    AddReward(StraightSpeedReward * speedRatio);
                }
            }
        }

        public override void CollectObservations(VectorSensor sensor)
        {
            sensor.AddObservation(m_Kart.LocalSpeed());

            var next = (m_CheckpointIndex + 1) % Colliders.Length;
            var nextCollider = Colliders[next];
            if (nextCollider == null)
                return;

            var direction = (nextCollider.transform.position - m_Kart.transform.position).normalized;
            sensor.AddObservation(Vector3.Dot(m_Kart.Rigidbody.velocity.normalized, direction));

            // Add observation for whether we're on a straight segment
            sensor.AddObservation(IsOnStraightSegment() ? 1f : 0f);

            if (ShowRaycasts)
                Debug.DrawLine(AgentSensorTransform.position, nextCollider.transform.position, Color.magenta);

            m_LastAccumulatedReward = 0.0f;
            m_EndEpisode = false;
            for (var i = 0; i < Sensors.Length; i++)
            {
                var current = Sensors[i];
                var xform = current.Transform;
                var hit = Physics.Raycast(AgentSensorTransform.position, xform.forward, out var hitInfo,
                    current.RayDistance, Mask, QueryTriggerInteraction.Ignore);

                if (ShowRaycasts)
                {
                    Debug.DrawRay(AgentSensorTransform.position, xform.forward * current.RayDistance, Color.green);
                    Debug.DrawRay(AgentSensorTransform.position, xform.forward * current.HitValidationDistance,
                        Color.red);

                    if (hit && hitInfo.distance < current.HitValidationDistance)
                    {
                        Debug.DrawRay(hitInfo.point, Vector3.up * 3.0f, Color.blue);
                    }
                }

                if (hit)
                {
                    if (hitInfo.distance < current.HitValidationDistance * 0.5f)
                    {
                        m_LastAccumulatedReward += HitPenalty;
                        m_EndEpisode = true;
                    }
                }

                sensor.AddObservation(hit ? hitInfo.distance : current.RayDistance);
            }

            sensor.AddObservation(m_Acceleration);
        }

        public override void OnActionReceived(ActionBuffers actions)
        {
            base.OnActionReceived(actions);
            InterpretDiscreteActions(actions);

            var next = (m_CheckpointIndex + 1) % Colliders.Length;
            var nextCollider = Colliders[next];
            var direction = (nextCollider.transform.position - m_Kart.transform.position).normalized;
            var reward = Vector3.Dot(m_Kart.Rigidbody.velocity.normalized, direction);

            if (ShowRaycasts) Debug.DrawRay(AgentSensorTransform.position, m_Kart.Rigidbody.velocity, Color.blue);

            // Basic rewards
            AddReward(reward * TowardsCheckpointReward);
            AddReward((m_Acceleration && !m_Brake ? 1.0f : 0.0f) * AccelerationReward);
            
            // Enhanced speed rewards with thresholds
            CalculateSpeedRewards();

            // Natural driving rewards
            AddSmoothSteeringReward();
            AddStraightSegmentRewards();
        }

        public override void OnEpisodeBegin()
        {
            switch (Mode)
            {
                case AgentMode.Training:
                    m_CheckpointIndex = Random.Range(0, Colliders.Length - 1);
                    var collider = Colliders[m_CheckpointIndex];
                    transform.localRotation = collider.transform.rotation;
                    transform.position = collider.transform.position;
                    m_Kart.Rigidbody.velocity = default;
                    m_Acceleration = false;
                    m_Brake = false;
                    m_Steering = 0f;
                    m_LastSteering = 0f; // Reset natural driving variables
                    break;
                default:
                    break;
            }
        }

        void InterpretDiscreteActions(ActionBuffers actions)
        {
            if (enableTimeScaleAdjustment)
            {
                // Apply separate multipliers for steering and throttle
                m_Steering = (actions.DiscreteActions[0] - 1f) * steeringMultiplier;
                m_Acceleration = actions.DiscreteActions[1] >= 1.0f;
                m_Brake = actions.DiscreteActions[1] < 1.0f;
            }
            else
            {
                // Normal action interpretation without scaling
                m_Steering = actions.DiscreteActions[0] - 1f;
                m_Acceleration = actions.DiscreteActions[1] >= 1.0f;
                m_Brake = actions.DiscreteActions[1] < 1.0f;
            }
        }

        public InputData GenerateInput()
        {
            return new InputData
            {
                Accelerate = m_Acceleration,
                Brake = m_Brake,
                TurnInput = m_Steering
            };
        }
    }
}
