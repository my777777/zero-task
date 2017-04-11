/*
 * ZeroLogger.h
 *
 *  Created on: Nov 24, 2016
 *      Author: zys
 */

#ifndef ZEROLOGGER_H_
#define ZEROLOGGER_H_
#include "Resource.h"

class ZeroLogger {
public:
	ZeroLogger();

	virtual ~ZeroLogger();

	static void LogIfInfo(bool printFlag,double* deviceData,int rows,int cols,string infoHead);

	static void LogIfInfoHost(bool printFlag,double *hostData, int rows, int cols, string infoHead);

};


#endif /* ZEROLOGGER_H_ */
