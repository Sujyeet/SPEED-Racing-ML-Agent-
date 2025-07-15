using KartGame.KartSystems;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
using UnityEngine;
using Random = UnityEngine.Random;
using System.Collections;

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
        public float MinSpeedThreshold = 3.5f;
        [Tooltip("Maximum speed for speed rewards")]
        public float MaxSpeedThreshold = 10f;
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
        
        public float OptimalStraightSpeed = 8f;
        [Tooltip("Minimum speed for straight segments")]
        public float MinStraightSpeed = 4f;
        [Tooltip("Angle threshold to consider a segment as 'straight'")]
        public float StraightAngleThreshold = 15f;
        #endregion

        #region Tire Obstacle Detection
        [Header("Tire Obstacle Settings")]
        [Tooltip("Layer mask specifically for tire obstacles (assign tires to this layer)")]
        public LayerMask TireObstacleMask = 1 << 9; // Layer 9 for tire obstacles
        [Tooltip("Reward for successfully navigating around tire obstacles")]
        public float TireAvoidanceReward = 0.012f;
        [Tooltip("Penalty for hitting tire obstacles")]
        public float TireHitPenalty = -0.25f;
        [Tooltip("Distance for tire avoidance reward")]
        public float TireAvoidanceDistance = 2.8f;
        [Tooltip("Speed penalty when near multiple tires")]
        public float NearTireSpeedPenalty = -0.004f;
        [Tooltip("Bonus reward for navigating through tire chicanes")]
        public float ChicaneNavigationBonus = 0.02f;
        [Tooltip("Enable tire obstacle detection")]
        public bool enableTireDetection = true;
        [Tooltip("Minimum distance between tires to consider them a chicane")]
        public float ChicaneDetectionDistance = 5f;
        #endregion

        #region Advanced Tire Navigation
        [Header("Advanced Tire Navigation")]
        [Tooltip("Reward for optimal path selection around tire clusters")]
        public float OptimalPathReward = 0.015f;
        [Tooltip("Penalty for taking suboptimal routes around tires")]
        public float SuboptimalPathPenalty = -0.008f;
        [Tooltip("Reward for maintaining racing line while avoiding tires")]
        public float RacingLineMaintenanceReward = 0.01f;
        [Tooltip("Distance threshold for optimal path detection")]
        public float OptimalPathThreshold = 1.5f;
        #endregion

        #region Recovery System
        [Header("Recovery System")]
        [Tooltip("Force magnitude for recovery assistance")]
        public float RecoveryForce = 300f;
        [Tooltip("Duration of recovery mode in seconds")]
        public float RecoveryDuration = 1.0f;
        [Tooltip("Speed threshold to trigger recovery mode")]
        public float RecoverySpeedThreshold = 1.0f;
        [Tooltip("Enable enhanced recovery system")]
        public bool enableEnhancedRecovery = true;
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

        // Tire obstacle tracking
        private int m_TireCollisionCount = 0;
        private float m_TotalTireAvoidanceReward = 0f;
        private int m_ChicaneNavigationCount = 0;
        private Vector3 m_LastValidPosition;

        // Enhanced recovery system variables
        private bool m_InRecoveryMode = false;
        private float m_RecoveryStartTime = 0f;
        private bool m_OriginalAcceleration = false;
        private bool m_OriginalBrake = false;
        private float m_OriginalSteering = 0f;

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
            
            // Initialize tracking variables
            m_LastValidPosition = transform.position;
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

            // Handle recovery mode timeout
            if (m_InRecoveryMode && Time.time - m_RecoveryStartTime > RecoveryDuration)
            {
                ExitRecoveryMode();
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

        // Enhanced recovery system
        void EnterRecoveryMode()
        {
            if (m_InRecoveryMode) return; // Already in recovery mode

            m_InRecoveryMode = true;
            m_RecoveryStartTime = Time.time;

            // Store original input states
            m_OriginalAcceleration = m_Acceleration;
            m_OriginalBrake = m_Brake;
            m_OriginalSteering = m_Steering;

            Debug.Log("Entering recovery mode - inputs overridden");
        }

        void ExitRecoveryMode()
        {
            if (!m_InRecoveryMode) return;

            m_InRecoveryMode = false;
            Debug.Log("Exiting recovery mode - normal control restored");
        }

        void ApplyRecoveryForces()
        {
            if (!m_InRecoveryMode) return;

            // Stop all current movement
            m_Kart.Rigidbody.velocity = Vector3.Lerp(m_Kart.Rigidbody.velocity, Vector3.zero, Time.deltaTime * 2f);
            m_Kart.Rigidbody.angularVelocity = Vector3.Lerp(m_Kart.Rigidbody.angularVelocity, Vector3.zero, Time.deltaTime * 3f);

            // Apply recovery forces
            Vector3 recoveryForce = -transform.forward * RecoveryForce;
            
            // Add random sideways component to avoid getting stuck in corners
            Vector3 sideForce = transform.right * UnityEngine.Random.Range(-100f, 100f);
            
            // Optional: Add slight upward force to clear obstacle geometry
            Vector3 upwardForce = Vector3.up * 100f;

            m_Kart.Rigidbody.AddForce(recoveryForce + sideForce + upwardForce);

            Debug.Log($"Recovery forces applied: Speed={m_Kart.LocalSpeed():F2}, Time={Time.time - m_RecoveryStartTime:F2}s");
        }

        // Comprehensive tire obstacle detection and navigation system
        void DetectTireObstacles()
        {
            if (!enableTireDetection) return;

            float nearestTireDistance = float.MaxValue;
            bool tireDetected = false;
            int tiresInRange = 0;
            Vector3[] tirePositions = new Vector3[10]; // Store up to 10 tire positions
            int tireCount = 0;
            
            for (var i = 0; i < Sensors.Length; i++)
            {
                var current = Sensors[i];
                var xform = current.Transform;
                
                // Specific raycast for tire detection
                var tireHit = Physics.Raycast(AgentSensorTransform.position, xform.forward, 
                    out var tireInfo, current.RayDistance, TireObstacleMask, QueryTriggerInteraction.Ignore);
                
                if (tireHit)
                {
                    tireDetected = true;
                    tiresInRange++;
                    nearestTireDistance = Mathf.Min(nearestTireDistance, tireInfo.distance);
                    
                    // Store tire position for chicane detection
                    if (tireCount < tirePositions.Length)
                    {
                        tirePositions[tireCount] = tireInfo.point;
                        tireCount++;
                    }
                    
                    if (ShowRaycasts)
                    {
                        Debug.DrawRay(tireInfo.point, Vector3.up * 2f, Color.yellow);
                        Debug.DrawLine(AgentSensorTransform.position, tireInfo.point, Color.blue);
                    }
                    
                    // Enhanced collision detection with improved recovery assistance
                    if (tireInfo.distance < current.HitValidationDistance)
                    {
                        AddReward(TireHitPenalty);
                        m_TireCollisionCount++;
                        Debug.Log($"Tire collision detected at distance: {tireInfo.distance}");
                        
                        // Enhanced recovery assistance for learning
                        if (enableEnhancedRecovery && m_Kart.LocalSpeed() < RecoverySpeedThreshold && !m_InRecoveryMode)
                        {
                            EnterRecoveryMode();
                        }
                    }
                }
            }
            
            // Apply recovery forces if in recovery mode
            if (m_InRecoveryMode)
            {
                ApplyRecoveryForces();
            }
            
            // Advanced tire navigation rewards
            if (tireDetected)
            {
                // Basic avoidance reward
                if (nearestTireDistance > TireAvoidanceDistance)
                {
                    float avoidanceQuality = Mathf.Clamp01((nearestTireDistance - TireAvoidanceDistance) / TireAvoidanceDistance);
                    float reward = TireAvoidanceReward * avoidanceQuality;
                    AddReward(reward);
                    m_TotalTireAvoidanceReward += reward;
                }
                
                // Speed management near tires
                float currentSpeed = m_Kart.LocalSpeed();
                if (nearestTireDistance < TireAvoidanceDistance && currentSpeed > OptimalStraightSpeed * 0.7f)
                {
                    AddReward(NearTireSpeedPenalty);
                }
                
                // Chicane navigation detection and reward
                if (DetectChicanePattern(tirePositions, tireCount))
                {
                    if (nearestTireDistance > TireAvoidanceDistance)
                    {
                        AddReward(ChicaneNavigationBonus);
                        m_ChicaneNavigationCount++;
                    }
                }
                
                // Optimal path reward for tire clusters
                if (tiresInRange >= 2)
                {
                    float pathOptimality = CalculatePathOptimality(tirePositions, tireCount);
                    if (pathOptimality > 0.7f)
                    {
                        AddReward(OptimalPathReward * pathOptimality);
                    }
                    else if (pathOptimality < 0.3f)
                    {
                        AddReward(SuboptimalPathPenalty);
                    }
                }
                
                // Racing line maintenance reward
                if (IsOnOptimalRacingLine() && nearestTireDistance > TireAvoidanceDistance)
                {
                    AddReward(RacingLineMaintenanceReward);
                }
            }
            
            // Update last valid position if not near obstacles
            if (!tireDetected || nearestTireDistance > TireAvoidanceDistance)
            {
                m_LastValidPosition = transform.position;
            }
        }

        // Detect chicane patterns in tire placement
        bool DetectChicanePattern(Vector3[] tirePositions, int count)
        {
            if (count < 2) return false;
            
            for (int i = 0; i < count - 1; i++)
            {
                for (int j = i + 1; j < count; j++)
                {
                    float distance = Vector3.Distance(tirePositions[i], tirePositions[j]);
                    if (distance > 2f && distance < ChicaneDetectionDistance)
                    {
                        // Check if tires are positioned to create a chicane (alternating pattern)
                        Vector3 midpoint = (tirePositions[i] + tirePositions[j]) / 2f;
                        Vector3 toMidpoint = (midpoint - transform.position).normalized;
                        Vector3 forward = transform.forward;
                        
                        if (Vector3.Dot(toMidpoint, forward) > 0.5f)
                        {
                            return true;
                        }
                    }
                }
            }
            return false;
        }

        // Calculate path optimality around tire obstacles
        float CalculatePathOptimality(Vector3[] tirePositions, int count)
        {
            if (count == 0) return 1f;
            
            Vector3 currentPosition = transform.position;
            Vector3 targetDirection = GetTargetDirection();
            
            float totalDistance = 0f;
            float optimalDistance = 0f;
            
            for (int i = 0; i < count; i++)
            {
                Vector3 toTire = tirePositions[i] - currentPosition;
                float distance = toTire.magnitude;
                totalDistance += distance;
                
                // Calculate optimal avoidance distance
                float dotProduct = Vector3.Dot(toTire.normalized, targetDirection);
                if (dotProduct > 0) // Tire is ahead
                {
                    optimalDistance += Mathf.Max(TireAvoidanceDistance, distance);
                }
            }
            
            if (totalDistance == 0f) return 1f;
            return Mathf.Clamp01(optimalDistance / totalDistance);
        }

        // Check if agent is maintaining optimal racing line
        bool IsOnOptimalRacingLine()
        {
            if (Colliders.Length < 2) return true;
            
            var next = (m_CheckpointIndex + 1) % Colliders.Length;
            var nextCollider = Colliders[next];
            
            Vector3 toCheckpoint = nextCollider.transform.position - transform.position;
            Vector3 forward = transform.forward;
            
            float alignment = Vector3.Dot(toCheckpoint.normalized, forward);
            return alignment > 0.8f; // Agent is well-aligned with racing line
        }

        // Get target direction for path optimization
        Vector3 GetTargetDirection()
        {
            if (Colliders.Length == 0) return transform.forward;
            
            var next = (m_CheckpointIndex + 1) % Colliders.Length;
            var nextCollider = Colliders[next];
            
            return (nextCollider.transform.position - transform.position).normalized;
        }

        // Add tire-specific observations to help agent make informed decisions
        void AddTireObservations(VectorSensor sensor)
        {
            if (!enableTireDetection) 
            {
                // Add default observations if tire detection is disabled
                for (int i = 0; i < Sensors.Length; i++)
                {
                    sensor.AddObservation(1f); // Max distance normalized
                }
                sensor.AddObservation(1f); // Nearest tire distance
                sensor.AddObservation(0f); // Tire density
                sensor.AddObservation(0f); // Danger zone
                sensor.AddObservation(0f); // Chicane detected
                return;
            }

            // Tire detection observations
            float[] tireDistances = new float[Sensors.Length];
            float nearestTireDistance = float.MaxValue;
            int tireCount = 0;
            bool chicaneDetected = false;
            Vector3[] tirePositions = new Vector3[10];
            int detectedTires = 0;
            
            for (var i = 0; i < Sensors.Length; i++)
            {
                var current = Sensors[i];
                var xform = current.Transform;
                
                var tireHit = Physics.Raycast(AgentSensorTransform.position, xform.forward, 
                    out var tireInfo, current.RayDistance, TireObstacleMask, QueryTriggerInteraction.Ignore);
                
                if (tireHit)
                {
                    tireDistances[i] = tireInfo.distance;
                    nearestTireDistance = Mathf.Min(nearestTireDistance, tireInfo.distance);
                    tireCount++;
                    
                    if (detectedTires < tirePositions.Length)
                    {
                        tirePositions[detectedTires] = tireInfo.point;
                        detectedTires++;
                    }
                }
                else
                {
                    tireDistances[i] = current.RayDistance; // Max distance if no tire
                }
            }
            
            // Check for chicane pattern
            chicaneDetected = DetectChicanePattern(tirePositions, detectedTires);
            
            // Add normalized tire distances
            foreach (float distance in tireDistances)
            {
                sensor.AddObservation(distance / Sensors[0].RayDistance);
            }
            
            // Add contextual tire information
            sensor.AddObservation(nearestTireDistance / Sensors[0].RayDistance); // Nearest tire
            sensor.AddObservation(tireCount / (float)Sensors.Length); // Tire density
            sensor.AddObservation(nearestTireDistance < TireAvoidanceDistance ? 1f : 0f); // Danger zone
            sensor.AddObservation(chicaneDetected ? 1f : 0f); // Chicane pattern detected
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

            // Add tire-specific observations
            AddTireObservations(sensor);

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
            
            // Tire obstacle detection and navigation
            DetectTireObstacles();
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
                    
                    // Reset tire tracking variables
                    m_TireCollisionCount = 0;
                    m_TotalTireAvoidanceReward = 0f;
                    m_ChicaneNavigationCount = 0;
                    m_LastValidPosition = transform.position;

                    // Reset recovery system
                    m_InRecoveryMode = false;
                    m_RecoveryStartTime = 0f;
                    break;
                default:
                    break;
            }
        }

        void InterpretDiscreteActions(ActionBuffers actions)
        {
            // Skip normal input interpretation if in recovery mode
            if (m_InRecoveryMode)
            {
                // Override inputs during recovery
                m_Acceleration = false;
                m_Brake = true;
                m_Steering = 0f; // Neutral steering during recovery
                return;
            }

            // Normal action interpretation
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
