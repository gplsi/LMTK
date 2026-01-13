---
number: 37
title: "Add multi-node support."
state: open
labels:
- enhancement
---

Here’s a well-structured GitHub issue you can use:

***

### **Title:** Add Multi-Node Support for Continual Pretraining

***

#### **Description**

Currently, continual pretraining workflows are limited to single-node setups, which restricts scalability and efficiency for large-scale models and datasets. Introducing multi-node support would enable distributed training across multiple machines, significantly improving throughput and reducing time-to-train.

***

#### **Motivation**

*   **Scalability:** As models and datasets grow, single-node training becomes a bottleneck.
*   **Efficiency:** Multi-node setups allow better resource utilization and faster iteration cycles.
*   **Flexibility:** Supports diverse hardware environments (e.g., clusters, cloud-based solutions).

***

#### **Proposed Solution**

*   Implement distributed data parallelism (DDP) or model parallelism for continual pretraining.
*   Integrate with existing frameworks like **PyTorch Distributed**, **DeepSpeed**, or **Horovod**.
*   Ensure compatibility with checkpointing and resuming workflows across nodes.

***

#### **Key Considerations**

*   **Fault Tolerance:** Handle node failures gracefully.
*   **Synchronization:** Efficient gradient synchronization to minimize communication overhead.
*   **Configuration:** Provide user-friendly options for cluster setup and resource allocation.

***

#### **Benefits**

*   Faster pretraining cycles for large-scale models.
*   Improved resource utilization in multi-GPU/multi-node environments.
*   Enables research and production teams to scale experiments seamlessly.

