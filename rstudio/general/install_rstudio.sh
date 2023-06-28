#!/bin/bash

set -e

apt-get update
apt-get install -y --no-install-recommends \
    file \
    git \
    libapparmor1 \
    libgc1c2 \
    libclang-dev \
    libcurl4-openssl-dev \
    libedit2 \
    libobjc4 \
    libssl-dev \
    libpq5 \
    lsb-release \
    psmisc \
    procps \
    python-setuptools \
    pwgen \
    sudo \
    wget

rm -rf /var/lib/apt/lists/*

# install s6 supervisor
/scripts/install_s6init.sh

export PATH=/usr/lib/rstudio-server/bin:$PATH

DOWNLOAD_FILE=rstudio-server.deb

wget https://download2.rstudio.org/server/bionic/amd64/rstudio-server-${RSTUDIO_VERSION/"+"/"-"}-amd64.deb -O "$DOWNLOAD_FILE"
dpkg -i "$DOWNLOAD_FILE"
rm "$DOWNLOAD_FILE"

# https://github.com/rocker-org/rocker-versioned2/issues/137
rm -f /var/lib/rstudio-server/secure-cookie-key

## RStudio wants an /etc/R, will populate from $R_HOME/etc
mkdir -p /etc/R
echo "PATH=${PATH}" >> ${R_HOME}/etc/Renviron.site

R_BIN=$(which R)
echo "rsession-which-r=${R_BIN}" > /etc/rstudio/rserver.conf
## use more robust file locking to avoid errors when using shared volumes:
echo "lock-type=advisory" > /etc/rstudio/file-locks

cp /etc/rstudio/rserver.conf /etc/rstudio/disable_auth_rserver.conf
echo "auth-none=1" >> /etc/rstudio/disable_auth_rserver.conf

## Set up RStudio init scripts
mkdir -p /etc/services.d/rstudio
# shellcheck disable=SC2016
echo '#!/usr/bin/with-contenv bash
## load /etc/environment vars first:
for line in $( cat /etc/environment ) ; do export $line > /dev/null; done
exec /usr/lib/rstudio-server/bin/rserver --server-daemonize 0' \
> /etc/services.d/rstudio/run

echo '#!/bin/bash
rstudio-server stop' \
> /etc/services.d/rstudio/finish

# If CUDA enabled, make sure RStudio knows (config_cuda_R.sh handles this anyway)
if [ ! -z "$CUDA_HOME" ]; then
  sed -i '/^rsession-ld-library-path/d' /etc/rstudio/rserver.conf
  echo "rsession-ld-library-path=$LD_LIBRARY_PATH" >> /etc/rstudio/rserver.conf
fi

# Log to stderr
LOGGING="[*]
log-level=warn
logger-type=syslog
"

printf "%s" "$LOGGING" > /etc/rstudio/logging.conf

# install user config initiation script
cp /scripts/init_userconf.sh /etc/cont-init.d/02_userconf
cp /scripts/pam-helper.sh /usr/lib/rstudio-server/bin/pam-helper

git config --system credential.helper 'cache --timeout=3600'
