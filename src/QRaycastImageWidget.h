
#ifndef __Q_RAYCAST_IMAGE_WIDGET_H__
#define __Q_RAYCAST_IMAGE_WIDGET_H__

#include "QImageWidget.h"
#include <QQuaternion>
#include <QVector2D>
#include <QBasicTimer>
#include <QMatrix4x4>

class QRaycastImageWidget : public QImageWidget
{
public:
	explicit QRaycastImageWidget(QWidget *parent = 0);
	~QRaycastImageWidget();

	void setWeelSpeed(float weel_speed){ weelSpeed = weel_speed; }
	void setPosition(float x, float y, float z){ position = QVector3D(x, y, z); }

	void setPerspective(float fov_y, float near_plane, float far_plane)
	{ 
		fovy = fov_y;
		nearPlane = near_plane;
		farPlane = far_plane;
	}

	void setup(const std::string& filepath, ushort vx_count, ushort vx_size);

public slots:
	virtual void setImage(const QImage& image);

protected:
	virtual void paintEvent(QPaintEvent *) Q_DECL_OVERRIDE;
	virtual void keyReleaseEvent(QKeyEvent *) Q_DECL_OVERRIDE;
	virtual void mousePressEvent(QMouseEvent *e) Q_DECL_OVERRIDE;
	virtual void mouseMoveEvent(QMouseEvent *e) Q_DECL_OVERRIDE;
	virtual void mouseReleaseEvent(QMouseEvent *e) Q_DECL_OVERRIDE;
	virtual void wheelEvent(QWheelEvent* e) Q_DECL_OVERRIDE;
	virtual void timerEvent(QTimerEvent *e) Q_DECL_OVERRIDE;
	virtual void resizeEvent(QResizeEvent *event) Q_DECL_OVERRIDE;

	void computeRaycast();
	void cleanup();

	QBasicTimer timer;
	QVector2D mousePressPosition;
	QVector3D rotationAxis;
	qreal angularSpeed;
	QQuaternion rotation;
	QVector3D position;
	float weelSpeed;
	float fovy;
	float nearPlane;
	float farPlane;
	QMatrix4x4 projection;

	QImage raycastImage;
	ushort voxel_count;
	ushort voxel_size;
	int vol_size;
};




#endif // __Q_RAYCAST_IMAGE_WIDGET_H__
