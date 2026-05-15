"""GCP-domain corpus: 10 paragraphs across 5 topics, 2 per topic.

Ground-truth relevance labels are set here and must never be changed —
they are the source of truth for all MRR/NDCG/P@3 computations.
Relevance scale: 0 = irrelevant, 1 = partially relevant, 2 = highly relevant.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    """A single retrievable unit with a unique ID, text, and topic tag."""

    id: str
    text: str
    topic: str


# ---------------------------------------------------------------------------
# Corpus — do NOT modify after initial creation
# ---------------------------------------------------------------------------

CORPUS: list[Chunk] = [
    # ── Cloud Run autoscaling ──────────────────────────────────────────────
    Chunk(
        id="chunk_0",
        text=(
            "Cloud Run automatically scales the number of container instances to handle "
            "incoming requests. When traffic increases beyond the current capacity, Cloud Run "
            "provisions additional instances within seconds to prevent request queuing. "
            "The --max-instances flag caps the autoscaling ceiling so you can control costs "
            "during unexpected traffic spikes or peak load events."
        ),
        topic="cloud_run_autoscaling",
    ),
    Chunk(
        id="chunk_1",
        text=(
            "Cloud Run's concurrency setting controls how many simultaneous requests a single "
            "container instance accepts. Setting concurrency to 1 dedicates one instance per "
            "request, which is appropriate for CPU-intensive or stateful workloads. "
            "The minimum-instances option keeps warm replicas alive to eliminate cold-start "
            "latency during traffic bursts, ensuring predictable scale-up behaviour under peak load."
        ),
        topic="cloud_run_autoscaling",
    ),
    # ── BigQuery slot allocation ───────────────────────────────────────────
    Chunk(
        id="chunk_2",
        text=(
            "BigQuery uses slots as the unit of computational capacity for executing SQL queries. "
            "Each slot represents a virtual CPU employed by the BigQuery query engine. "
            "On-demand pricing charges per byte scanned, while slot reservations let teams commit "
            "to a fixed capacity regardless of actual utilisation, offering cost predictability "
            "for high-volume analytical workloads."
        ),
        topic="bigquery_slot_allocation",
    ),
    Chunk(
        id="chunk_3",
        text=(
            "BigQuery slot reservations allow organisations to purchase a guaranteed number of "
            "processing slots at a discounted rate compared to on-demand pricing. "
            "Commitment options of one year or three years reduce the per-slot cost further. "
            "Unused reserved slots can be shared across projects within the same organisation "
            "via slot reservation assignments, maximising resource utilisation across teams."
        ),
        topic="bigquery_slot_allocation",
    ),
    # ── Pub/Sub flow control ───────────────────────────────────────────────
    Chunk(
        id="chunk_4",
        text=(
            "Cloud Pub/Sub flow control prevents subscriber clients from being overwhelmed by "
            "message delivery. The MaxOutstandingMessages setting limits the number of messages "
            "a subscriber can hold in memory simultaneously. When that limit is reached, the "
            "client automatically pauses message delivery until existing messages are acknowledged, "
            "applying backpressure to the upstream message pipeline."
        ),
        topic="pubsub_flow_control",
    ),
    Chunk(
        id="chunk_5",
        text=(
            "Pub/Sub flow control operates at the subscription level and uses lease management "
            "to coordinate delivery across multiple subscriber instances. "
            "The MaxOutstandingBytes setting bounds the total memory consumed by unacknowledged "
            "messages, complementing MaxOutstandingMessages. Together these flow control "
            "parameters ensure backpressure propagates upstream when consumers cannot keep "
            "pace with the publisher throughput rate."
        ),
        topic="pubsub_flow_control",
    ),
    # ── Vertex AI endpoint serving ─────────────────────────────────────────
    Chunk(
        id="chunk_6",
        text=(
            "Vertex AI Online Prediction endpoints provide low-latency, real-time model inference "
            "via a managed REST API. Models deployed to endpoints are served on dedicated compute "
            "instances with automatic load balancing across replicas. Traffic splitting enables "
            "gradual rollout of new model versions by routing a configurable percentage of "
            "prediction requests to each deployed model version, reducing serving risk."
        ),
        topic="vertex_ai_endpoint_serving",
    ),
    Chunk(
        id="chunk_7",
        text=(
            "Vertex AI endpoints support autoscaling based on prediction request volume, scaling "
            "from zero replicas when idle to handle burst prediction traffic. "
            "Dedicated GPU and TPU accelerators reduce inference latency for large language "
            "and vision models deployed on Vertex AI. The target-replicas setting configures "
            "the desired number of model server instances, balancing serving latency against "
            "infrastructure cost."
        ),
        topic="vertex_ai_endpoint_serving",
    ),
    # ── GKE cluster autoscaler ─────────────────────────────────────────────
    Chunk(
        id="chunk_8",
        text=(
            "The GKE cluster autoscaler automatically adjusts the number of nodes in a node pool "
            "based on workload resource demands. When pods cannot be scheduled because the cluster "
            "has insufficient CPU or memory, the autoscaler provisions new nodes to accommodate "
            "the pending workload. During sustained low utilisation, the cluster autoscaler "
            "removes underutilised nodes to reduce infrastructure costs."
        ),
        topic="gke_cluster_autoscaler",
    ),
    Chunk(
        id="chunk_9",
        text=(
            "GKE node auto-provisioning extends the cluster autoscaler by automatically creating "
            "and deleting entire node pools based on the resource requirements of pending pods. "
            "It selects the optimal machine type and GPU/TPU accelerator configuration for each "
            "workload's needs. Vertical Pod Autoscaler works alongside the cluster autoscaler to "
            "right-size individual pod CPU and memory requests, improving cluster packing density."
        ),
        topic="gke_cluster_autoscaler",
    ),
]
