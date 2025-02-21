import time
import math
import torch
from tqdm import tqdm

class BaseTrainer:
    def __init__(self, fabric, model, optimizer, scheduler, train_dataloader, validation_dataloader, config, monitor):
        self.fabric = fabric
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader
        self.config = config
        self.monitor = monitor
        self.state = {
            "model": model,
            "optimizer": optimizer,
            "iter_num": 0,
            "step_count": 0,
            "scheduler": scheduler,
        }

    def train(self, resume=False):
        total_t0 = time.perf_counter()
        initial_iter = self.state["iter_num"]
        curr_iter = 0
        epochs = self.config.number_epochs

        for epoch in range(epochs):
            if self.fabric.global_rank == 0:
                print(f"Ejecutando Epoch {epoch + 1} de {epochs}")
            batch_iterator = tqdm(self.train_dataloader, colour="blue") if self.fabric.global_rank == 0 else self.train_dataloader

            for step, batch in enumerate(batch_iterator):
                # Manejo de reanudación
                if resume and curr_iter < initial_iter:
                    curr_iter += 1
                    continue
                elif resume:
                    resume = False
                    curr_iter = -1
                    self.fabric.barrier()
                    self.fabric.print("Reanudación completada")
                    
                if self.state["iter_num"] >= self.config.max_iters:
                    break

                iter_t0 = time.perf_counter()
                is_accumulating = (self.state["iter_num"] + 1) % self.config.gradient_accumulation_steps != 0

                # Bloque genérico de forward y backward
                with self.fabric.no_backward_sync(self.model, enabled=is_accumulating):
                    outputs, loss = self.model.training_step(batch, step)
                    self.fabric.backward(loss / self.config.gradient_accumulation_steps)

                if not is_accumulating:
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                    self.fabric.print(f"Norma del gradiente: {grad_norm:.4f}")
                    self.fabric.clip_gradients(self.model, self.optimizer, max_norm=self.config.grad_clip)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    self.state["step_count"] += 1

                    if self.state["step_count"] % self.config.eval_steps == 0:
                        self.fabric.barrier()
                        val_loss = self.validate()
                        self.fabric.log_dict({"metric/val_loss": val_loss.item()}, self.state["step_count"])
                        self.fabric.log_dict({"metric/val_ppl": math.exp(val_loss.item())}, self.state["step_count"])
                        self.fabric.barrier()
                        ckpt_path = self.config.output_dir / f"iter-{self.state['iter_num']:06d}-ckpt.pth"
                        self.fabric.print(f"Guardando checkpoint en {str(ckpt_path)!r}")
                        self.fabric.save(ckpt_path, self.state)

                self.state["iter_num"] += 1
                t1 = time.perf_counter()
                self.fabric.print(
                    f"iter {self.state['iter_num']} step {self.state['step_count']}: loss {loss.item():.4f}, tiempo: {(t1 - iter_t0) * 1000:.2f}ms"
                )
                self.monitor.on_train_batch_end(
                    self.state["iter_num"] * self.config.micro_batch_size,
                    t1 - total_t0,
                    self.fabric.world_size,
                    self.state["step_count"],
                    train_loss=loss.item()
                )

            self.fabric.barrier()
            val_loss = self.validate()
            self.fabric.print(f"Epoch {epoch + 1}: val loss {val_loss:.4f}")
            ckpt_path = self.config.output_dir / f"iter-{self.state['iter_num']:06d}-ckpt.pth"
            self.fabric.print(f"Guardando checkpoint en {str(ckpt_path)!r}")
            self.fabric.save(ckpt_path, self.state)

    @torch.no_grad()
    def validate(self) -> torch.Tensor:
        self.fabric.print("Validando...")
        self.model.eval()
        losses = torch.zeros(self.config.eval_iters, device=self.fabric.device)
        batch_iterator = tqdm(self.validation_dataloader, desc="Validación", colour="green")
        for k, batch in enumerate(batch_iterator):
            if k >= self.config.eval_iters:
                break
            outputs = self.model.validation_step(batch, k)
            losses[k] = outputs.loss
        self.model.train()
        return losses.mean()



class TrainerFSDP(BaseTrainer):
    def train(self, resume=False):
        self.fabric.print("Iniciando entrenamiento con FSDP")
        
        # Si es necesario, se pueden incluir aquí pasos adicionales previos al entrenamiento,
        # específicos de FSDP, por ejemplo, verificaciones o configuraciones especiales.
        # Por ejemplo: asegurarse de que la estrategia FSDP esté correctamente inicializada.
        
        super().train(resume=resume)
    
    # Si FSDP requiere algún cambio en la validación (por ejemplo, consolidar parámetros de
    # distintos procesos antes de calcular la métrica), se puede sobrescribir este método.
    # En este ejemplo, mantenemos el comportamiento genérico.
    # def validate(self) -> torch.Tensor:
    #     # Paso extra: sincronización o gathering de estados si fuera necesario.
    #     return super().validate()
