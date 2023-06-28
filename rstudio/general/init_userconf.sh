#!/bin/bash

USER=${USER}
USERID=${USERID}
GROUPID=${GROUPID}
ROOT=${ROOT:=FALSE}
UMASK=${UMASK:=022}
TZ=${TZ:=Etc/UTC}

if [[ ${DISABLE_AUTH,,} == "true" ]]
then
	mv /etc/rstudio/disable_auth_rserver.conf /etc/rstudio/rserver.conf
	echo "USER=$USER" >> /etc/environment
fi

if grep --quiet "auth-none=1" /etc/rstudio/rserver.conf
then
	echo "Skipping authentication as requested"
fi

if [[ "$USERID" -lt 1000 ]]
# Probably a macOS user, https://github.com/rocker-org/rocker/issues/205
  then
    echo "$USERID is less than 1000"
    check_user_id=$(grep -F "auth-minimum-user-id" /etc/rstudio/rserver.conf)
    if [[ ! -z $check_user_id ]]
    then
      echo "minumum authorised user already exists in /etc/rstudio/rserver.conf: $check_user_id"
    else
      echo "setting minumum authorised user to 499"
      echo auth-minimum-user-id=499 >> /etc/rstudio/rserver.conf
    fi
fi

if [[ "$USERID" -ge 1000 ]]
## Configure user with a different USERID if requested.
  then
    mkdir -p /home/${USER}/.rstudio/monitored/user-settings
    echo "alwaysSaveHistory='0' \
      \nloadRData='0' \
      \nsaveAction='0'" \
      > /home/${USER}/.rstudio/monitored/user-settings/user-settings

    echo "creating new $USER with UID $USERID"
    useradd -m $USER -u $USERID
    mkdir -p /home/$USER
    chown -R $USER /home/$USER
    usermod -a -G staff $USER
fi

if [[ "$GROUPID" -ge 1000 ]]
## Configure the primary GID (whether rstudio or $USER) with a different GROUPID if requested.
  then
    echo "Modifying primary group $(id $USER -g -n)"
    groupmod -g $GROUPID $(id $USER -g -n)
    echo "Primary group ID is now custom_group $GROUPID"
fi

## Add a password to user
#echo "$USER:$PASSWORD" | chpasswd

# Use Env flag to know if user should be added to sudoers
if [[ ${ROOT,,} == "true" ]]
  then
    adduser $USER sudo && echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
    echo "$USER added to sudoers"
fi

## Change Umask value if desired
if [ "$UMASK" -ne 022 ]
  then
    echo "server-set-umask=false" >> /etc/rstudio/rserver.conf
    echo "Sys.umask(mode=$UMASK)" >> /home/$USER/.Rprofile
fi

## Next one for timezone setup
if [ "$TZ" !=  "Etc/UTC" ]
  then
    ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
fi

