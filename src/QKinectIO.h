#ifndef __Q_KINECT_STREAM_H__
#define __Q_KINECT_STREAM_H__

#include <QObject>
#include "QKinectReader.h"

class QKinectIO
{

public:
	QKinectIO();
	virtual ~QKinectIO();

	void setKinecReader(QKinectReader* kinect);
	
	void setMaxSizeInMegaBytes(unsigned int size_in_Mbytes);

	void appendFrame();
	void save(const QString& filename);
	void load(const QString& filename);

	void clear();
	unsigned int size() const;
	

private:

	static void saveFile(QString filename, std::vector<unsigned short> info, std::vector<unsigned short> depthBuffer);
	static void loadFile(QString filename, std::vector<unsigned short>& info, std::vector<unsigned short>& depthBuffer);


	QKinectReader*				kinectReader;
	std::vector<unsigned short> depthBufferStream;
	unsigned int				maxSizeInMegaBytes;
};




#endif	//__Q_KINECT_STREAM_H__


