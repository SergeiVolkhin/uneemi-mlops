variable "yc_token" {
  description = "OAuth/IAM-токен Yandex Cloud"
  type        = string
  sensitive   = true
}

variable "yc_cloud_id" {
  description = "ID облака Yandex Cloud"
  type        = string
}

variable "yc_folder_id" {
  description = "ID каталога Yandex Cloud"
  type        = string
}

variable "yc_zone" {
  description = "Зона доступности"
  type        = string
  default     = "ru-central1-a"
}

variable "vm_cores" {
  description = "Число vCPU (стек ~11 контейнеров)"
  type        = number
  default     = 4
}

variable "vm_memory_gb" {
  description = "ОЗУ, ГБ"
  type        = number
  default     = 8
}

variable "vm_disk_gb" {
  description = "Размер диска, ГБ"
  type        = number
  default     = 40
}

variable "ssh_public_key" {
  description = "Публичный SSH-ключ для доступа к VM"
  type        = string
}

variable "repo_url" {
  description = "URL git-репозитория со стеком (cloud-init его клонирует)"
  type        = string
}

variable "ssh_allowed_cidr" {
  description = "Откуда разрешён доступ (для защиты сузить до своего IP)"
  type        = list(string)
  default     = ["0.0.0.0/0"]
}

variable "allowed_ports" {
  description = "Host-порты сервисов стека + SSH"
  type        = list(number)
  # ssh, serving, airflow, grafana, mlflow, prometheus
  default     = [22, 18000, 18080, 13000, 5500, 19090]
}
