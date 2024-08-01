# !/bin/bash

SETUSER="svcapp"
RUNNER=`whoami`
export SERVER_NAME="gemini-long-text-imagen2"


if [ ${RUNNER} != ${SETUSER} ] ;
   then echo "Deny Access : [ ${RUNNER} ]. Not ${SETUSER}" ;
   exit 0 ;
fi

ps -ef| grep ${SERVER_NAME} | grep -v 'grep'| awk {'print "kill -9 " $2'} | sh -x


