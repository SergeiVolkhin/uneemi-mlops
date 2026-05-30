# Скелет облачного развёртывания Uneemi MLOps в Yandex Cloud.
# СТАТУС: скелет (открытый гейт 2-го балла критерия 3). Не применяется автоматически.
# Поднимает один VPS, ставит Docker и запускает весь стек через docker compose.
# Для реального деплоя: заполнить terraform.tfvars, затем terraform init && apply.

terraform {
  required_version = ">= 1.5"
  required_providers {
    yandex = {
      source  = "yandex-cloud/yandex"
      version = ">= 0.120"
    }
  }
}

provider "yandex" {
  token     = var.yc_token
  cloud_id  = var.yc_cloud_id
  folder_id = var.yc_folder_id
  zone      = var.yc_zone
}

# Сеть и подсеть.
resource "yandex_vpc_network" "uneemi" {
  name = "uneemi-net"
}

resource "yandex_vpc_subnet" "uneemi" {
  name           = "uneemi-subnet"
  zone           = var.yc_zone
  network_id     = yandex_vpc_network.uneemi.id
  v4_cidr_blocks = ["10.10.0.0/24"]
}

# Группа безопасности: SSH + host-порты сервисов стека (в обход занятых 6379/8000).
resource "yandex_vpc_security_group" "uneemi" {
  name       = "uneemi-sg"
  network_id = yandex_vpc_network.uneemi.id

  egress {
    protocol       = "ANY"
    v4_cidr_blocks = ["0.0.0.0/0"]
    from_port      = 0
    to_port        = 65535
  }

  dynamic "ingress" {
    for_each = var.allowed_ports
    content {
      protocol       = "TCP"
      v4_cidr_blocks = var.ssh_allowed_cidr
      port           = ingress.value
    }
  }
}

# Базовый образ Ubuntu 22.04 LTS.
data "yandex_compute_image" "ubuntu" {
  family = "ubuntu-2204-lts"
}

# Виртуальная машина со стеком. cloud-init ставит Docker и поднимает compose.
resource "yandex_compute_instance" "uneemi" {
  name        = "uneemi-mlops"
  platform_id = "standard-v3"
  zone        = var.yc_zone

  resources {
    cores  = var.vm_cores
    memory = var.vm_memory_gb
  }

  boot_disk {
    initialize_params {
      image_id = data.yandex_compute_image.ubuntu.id
      size     = var.vm_disk_gb
    }
  }

  network_interface {
    subnet_id          = yandex_vpc_subnet.uneemi.id
    nat                = true # внешний IP, чтобы сервис был доступен снаружи до защиты
    security_group_ids = [yandex_vpc_security_group.uneemi.id]
  }

  metadata = {
    user-data = templatefile("${path.module}/cloud-init.yaml", {
      repo_url       = var.repo_url
      repo_ref       = var.repo_ref
      ssh_public_key = var.ssh_public_key
    })
    ssh-keys = "ubuntu:${var.ssh_public_key}"
  }
}
