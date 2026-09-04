#!/bin/bash
# WebArena-Lite self-hosted sites.
#
# Pulls 4 of the 5 sites and SKIPS map. Map alone is ~180GB and 134 of the 165
# WebArena-Lite tasks need no map at all (and none need Wikipedia), so skipping
# it costs 31 tasks and saves more disk and time than everything else combined.
#
# Runs in parallel with the model build on the same box -- these are network and
# disk bound, the build is CPU bound.
set -ux
H=$HOME
W=$H/browse
mkdir -p "$W/logs"
exec > "$W/logs/sites.log" 2>&1
step () { echo "=== $(date '+%F %H:%M:%S') STEP: $* ==="; }

# The sites must be reachable at a hostname the browser inside the harness can
# resolve. Everything runs on this one box, so the private IP is correct.
IP=$(hostname -I | awk '{print $1}')
echo "HOST_IP=$IP" | tee "$W/HOST_IP"

step "docker available?"
docker --version || sudo apt-get install -y -qq docker.io
sudo systemctl start docker || true
sudo docker ps > /dev/null || die "docker not usable"

step "pull site images (skipping map: ~180GB for 31 tasks)"
# Smallest first so something is usable early.
for img in \
  webarenaimages/shopping_admin_final_0719 \
  webarenaimages/postmill-populated-exposed-withimg \
  webarenaimages/shopping_final_0712 \
  webarenaimages/gitlab-populated-final ; do
  echo "--- pulling $img ---"
  sudo docker pull "$img" || echo "WARN: pull failed for $img"
  df -h / | tail -1
done
sudo docker images | head -8
touch "$W/SITES_PULLED"
step "SITES PULLED"

step "start containers"
sudo docker run --name shopping_admin -p 7780:80 -d webarenaimages/shopping_admin_final_0719 || true
sudo docker run --name shopping      -p 7770:80 -d webarenaimages/shopping_final_0712 || true
sudo docker run --name forum         -p 9999:80 -d webarenaimages/postmill-populated-exposed-withimg || true
sudo docker run --name gitlab -d -p 8023:8023 webarenaimages/gitlab-populated-final \
  /opt/gitlab/embedded/bin/runsvdir-start || true
sleep 90
sudo docker ps --format '{{.Names}}\t{{.Status}}'

step "rewrite base URLs to this host (sites bake in absolute URLs)"
sudo docker exec shopping_admin /var/www/magento2/bin/magento setup:store-config:set --base-url="http://$IP:7780" || true
sudo docker exec shopping_admin mysql -u magentouser -pMyPassword magentodb -e \
  "UPDATE core_config_data SET value=\"http://$IP:7780/\" WHERE path = \"web/secure/base_url\";" || true
sudo docker exec shopping_admin php /var/www/magento2/bin/magento config:set admin/security/password_is_forced 0 || true
sudo docker exec shopping_admin php /var/www/magento2/bin/magento config:set admin/security/password_lifetime 0 || true
sudo docker exec shopping_admin /var/www/magento2/bin/magento cache:flush || true

sudo docker exec shopping /var/www/magento2/bin/magento setup:store-config:set --base-url="http://$IP:7770" || true
sudo docker exec shopping mysql -u magentouser -pMyPassword magentodb -e \
  "UPDATE core_config_data SET value=\"http://$IP:7770/\" WHERE path = \"web/secure/base_url\";" || true
sudo docker exec shopping /var/www/magento2/bin/magento cache:flush || true

step "gitlab reconfigure (slow, 5-10 min)"
sudo docker exec gitlab sed -i "s|^external_url.*|external_url 'http://$IP:8023'|" /etc/gitlab/gitlab.rb || true
sudo docker exec gitlab gitlab-ctl reconfigure || true

step "reachability check"
for p in 7770 7780 9999 8023; do
  code=$(curl -s -o /dev/null -w '%{http_code}' -m 20 "http://$IP:$p" || echo TIMEOUT)
  echo "port $p -> $code"
done
touch "$W/SITES_READY"
step "SITES READY"
