

#ifndef _Q_RAYCAST_IMAGE_WIDGET_H
#define _Q_RAYCAST_IMAGE_WIDGET_H

#include "QImageWidget.h"


class RaycastImageWidget : public QImageWidget
{
    Q_OBJECT

public:
	RaycastImageWidget(QWidget* parent = nullptr);
	virtual ~RaycastImageWidget();

public slots:

	void setup(const std::string& filepath, ushort vx_count, ushort vx_size);
	void cleanup();
	void computeRaycast();

Q_SIGNALS:
	void closed();

protected:
	virtual void keyReleaseEvent(QKeyEvent *) Q_DECL_OVERRIDE;
	virtual void wheelEvent(QWheelEvent* e) Q_DECL_OVERRIDE;


	QImage raycastImage;
	float cam_z_coord;
	float fov;
	ushort voxel_count;
	ushort voxel_size;
};

#endif	// _Q_RAYCAST_IMAGE_WIDGET_H
